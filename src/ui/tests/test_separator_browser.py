"""Real browser DOM verification; no screenshots or multimodal inspection."""
import socket
import threading
import time

from playwright.sync_api import expect, sync_playwright
import pytest
import uvicorn

from ui.separator_app import create_separator_app


@pytest.mark.parametrize('width',[390,1280])
def test_partner_predictions_planning_and_tables_in_text_browser(tmp_path,monkeypatch,width):
    def provider(*args,**kwargs):
        time.sleep(1.2)
        return {'model':'gpt-6-astra','result':{'conductivity_mS_cm':.7,
            'preparation_hypothesis':'此為機制假說，非觀察結果。BNNT 可能改善浸潤。','supporting_record_ids':[]}}
    monkeypatch.setattr('ui.separator_app.run_separator_model',provider)
    app=create_separator_app(runtime_dir=tmp_path,model_executable='/fake/codex')
    with socket.socket() as listener:
        listener.bind(('127.0.0.1',0))
        port=listener.getsockname()[1]
        server=uvicorn.Server(uvicorn.Config(app,log_level='error',access_log=False))
        thread=threading.Thread(target=server.run,kwargs={'sockets':[listener]},daemon=True)
        thread.start()
        try:
            deadline=time.monotonic()+10
            while not server.started and time.monotonic()<deadline:
                time.sleep(.02)
            assert server.started
            with sync_playwright() as p:
                browser=p.chromium.launch(channel='chrome',headless=True)
                try:
                    page=browser.new_page(viewport={'width':width,'height':844})
                    page.emulate_media(color_scheme='dark' if width==390 else 'light')
                    errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
                    page.goto(f'http://127.0.0.1:{port}/#access=obsolete',wait_until='networkidle')
                    page.locator('#record-table tbody tr').first.wait_for()
                    assert 'AI for Science' in page.title()
                    assert page.locator('#record-table tbody tr').count()==12
                    assert page.locator('html').get_attribute('data-theme')==('dark' if width==390 else 'light')
                    before=page.locator('body').evaluate('(el)=>getComputedStyle(el).backgroundColor')
                    page.locator('#theme-toggle').click()
                    after=page.locator('body').evaluate('(el)=>getComputedStyle(el).backgroundColor')
                    assert before!=after
                    page.reload(wait_until='networkidle')
                    assert page.locator('body').evaluate('(el)=>getComputedStyle(el).backgroundColor')==after
                    page.locator('#theme-toggle').focus();page.keyboard.press('Enter')
                    assert page.locator('body').evaluate('(el)=>getComputedStyle(el).backgroundColor')==before
                    assert page.locator('#access-code, pre, [data-case]').count()==0
                    assert '#' not in page.url
                    assert 'Astra' not in page.locator('body').inner_text()
                    assert '17.5%' in page.locator('#workflow-result').inner_text()
                    assert page.locator('#privacy, #hours-per-experiment, #time-saved, #workflow-context, #workflow-methods, #workflow-interval, #glossary').count()==0
                    assert '下一輪實驗推薦' in page.locator('#planning').inner_text()
                    assert page.locator('#sources, #source-list, a[href="/api/download/sources.csv"], a[href="/api/download/public.sqlite"]').count()==0
                    assert not any('？' in t for t in page.locator('h1,h2,h3').all_inner_texts())
                    page.locator('#baseline').select_option('nearest')
                    assert '0.8%' in page.locator('#workflow-result').inner_text()
                    page.locator('#aqueous-form button').click()
                    expect(page.locator('#prediction')).to_contain_text('7.45 mPa·s')
                    page.locator('#aqueous-form input').fill('12')
                    page.locator('#aqueous-form button').click()
                    expect(page.locator('#prediction')).to_contain_text('未達到 1.5 μm 塗層')
                    page.locator('#task').select_option('bnnt_conductivity')
                    page.locator('#bnnt-form button').click()
                    expect(page.locator('#prediction')).to_contain_text('20–60 秒')
                    expect(page.locator('#task')).to_be_disabled()
                    expect(page.locator('#prediction')).to_contain_text('0.803 mS/cm')
                    assert '非觀察結果' not in page.locator('#prediction').inner_text()
                    assert '可能改善' in page.locator('#prediction').inner_text()
                    page.locator('#task').select_option('coating_thickness')
                    page.locator('#coating-form button').click()
                    expect(page.locator('#prediction')).to_contain_text('119.5 μm')
                    page.locator('#task').select_option('electrolyte_conductivity')
                    page.locator('#electrolyte-form button').click()
                    expect(page.locator('#prediction .metric')).to_contain_text('mS/cm')
                    assert page.locator('#emc-percent').inner_text()=='10%'
                    page.locator('#plan-form button[type=submit]').click()
                    page.locator('#plan-result tbody tr').first.wait_for()
                    assert page.locator('#plan-result tbody tr').count()==1
                    page.locator('#catalogue-type').select_option('separator')
                    page.locator('#record-search').fill('CA@BN-3:1-failure')
                    assert page.locator('#record-table tbody tr').count()==1
                    assert '孔道堵塞' in page.locator('#record-table').inner_text()
                    page.locator('[data-record]').click()
                    page.locator('#record-details summary').click()
                    page.locator('[data-evidence]').first.click()
                    page.locator('#evidence-output .evidence').wait_for()
                    assert '3:1' in page.locator('#evidence-output').inner_text()
                    assert 'Preparation of the CA@BN Separator' in page.locator('#evidence-output').inner_text()
                    assert 'xml' not in page.locator('#evidence-output > p').all_inner_texts()[1].lower()
                    page.locator('#record-search').fill('')
                    page.locator('#catalogue-type').select_option('electrolyte')
                    assert page.locator('#record-table tbody tr').count()==38
                    assert 'EC' in page.locator('#record-table').inner_text()
                    assert errors==[]
                    assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
                finally:
                    browser.close()
        finally:
            server.should_exit=True
            thread.join(timeout=10)
            assert not thread.is_alive()
