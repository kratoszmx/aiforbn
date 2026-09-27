"""Real browser DOM verification; no screenshots or multimodal inspection."""
import socket
import threading
import time

from playwright.sync_api import sync_playwright
import uvicorn

from ui.separator_app import create_separator_app


def test_separator_partner_flow_in_text_browser(tmp_path):
    app=create_separator_app(runtime_dir=tmp_path)
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
                    page=browser.new_page(viewport={'width':390,'height':844})
                    errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
                    page.goto(f'http://127.0.0.1:{port}/#access=fixture-invitation',wait_until='networkidle')
                    page.locator('#record-list .card').first.wait_for()
                    assert page.locator('#record-list .card').count()==20
                    assert page.locator('#access-code').input_value()=='fixture-invitation'
                    assert '#' not in page.url
                    page.locator('[data-case="known"]').click()
                    assert page.locator('#access-code').input_value()=='fixture-invitation'
                    page.get_by_text('可以做探索性試算；尚未驗證準確度',exact=True).wait_for()
                    page.locator('[data-case="holdout"]').click()
                    assert page.locator('[name="loading_mg_cm2"]').input_value()=='.5'
                    page.locator('[data-case="unsupported"]').click()
                    assert page.locator('#access-code').input_value()=='fixture-invitation'
                    page.get_by_text('這組輸入暫不支持數值預測',exact=True).wait_for()
                    page.locator('#record-search').fill('CA@BN-3:1-failure')
                    page.wait_for_function("document.querySelectorAll('#record-list .card').length === 1")
                    assert 'failed_pore_clogging' in page.locator('#record-list').inner_text()
                    page.locator('#record-list summary').nth(1).click()
                    page.locator('.evidence-link').first.click()
                    page.locator('.evidence-output .evidence').wait_for()
                    assert '3:1' in page.locator('.evidence-output .evidence').inner_text()
                    assert errors==[]
                    assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
                finally:
                    browser.close()
        finally:
            server.should_exit=True
            thread.join(timeout=10)
            assert not thread.is_alive()
