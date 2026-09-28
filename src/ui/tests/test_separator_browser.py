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
    monkeypatch.setattr('ui.separator_app.run_separator_model',lambda *args,**kwargs:{
        'model':'gpt-6-astra','result':{'conductivity_mS_cm':.7,
        'preparation_hypothesis':'按公開配方估算。','supporting_record_ids':[]}})
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
                    errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
                    page.goto(f'http://127.0.0.1:{port}/#access=obsolete',wait_until='networkidle')
                    page.locator('#record-table tbody tr').first.wait_for()
                    assert 'AI for Science' in page.title()
                    assert page.locator('#record-table tbody tr').count()==20
                    assert page.locator('#access-code, pre, [data-case]').count()==0
                    assert '#' not in page.url
                    assert 'Astra' not in page.locator('body').inner_text()
                    assert '17.5%' in page.locator('#workflow-result').inner_text()
                    assert page.locator('#privacy, #hours-per-experiment, #time-saved, #workflow-context, #workflow-methods, #workflow-interval, #glossary').count()==0
                    assert '下一個配方，先試哪一個？' in page.locator('#planning').inner_text()
                    page.locator('#baseline').select_option('nearest')
                    assert '0.8%' in page.locator('#workflow-result').inner_text()
                    page.locator('#bnnt-form button').click()
                    expect(page.locator('#prediction')).to_contain_text('0.803 mS/cm')
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
                    page.locator('#record-search').fill('CA@BN-3:1-failure')
                    assert page.locator('#record-table tbody tr').count()==1
                    assert '孔道堵塞' in page.locator('#record-table').inner_text()
                    page.locator('[data-record]').click()
                    page.locator('#record-details summary').click()
                    page.locator('[data-evidence]').first.click()
                    page.locator('#evidence-output .evidence').wait_for()
                    assert '3:1' in page.locator('#evidence-output').inner_text()
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
