import os
from playwright.async_api import async_playwright
import asyncio


async def login_to_fidelity(username: str, password: str, use_default_profile: bool = False):
    """
    Login to Fidelity using Playwright

    Args:
        username: Your Fidelity username
        password: Your Fidelity password
        use_default_profile: If True, uses your default Chrome profile with saved cookies/sessions
    """
    async with async_playwright() as p:
        # Get default Chrome user data directory on Windows
        if use_default_profile:
            user_data_dir = os.path.expanduser(r"~\AppData\Local\Google\Chrome\User Data")
            browser = await p.chromium.launch_persistent_context(
                user_data_dir=user_data_dir,
                channel="chrome",  # Use system Chrome
                headless=False,
                args=["--disable-blink-features=AutomationControlled"]
            )
            page = browser.pages[0] if browser.pages else await browser.new_page()
        else:
            # Launch browser using system Chrome (default Windows Chrome)
            browser = await p.chromium.launch(
                channel="chrome",  # Use system Chrome instead of bundled Chromium
                headless=False,
                args=["--disable-blink-features=AutomationControlled"]  # Reduce automation detection
            )
            page = await browser.new_page()

        await page.goto("https://digital.fidelity.com/prgw/digital/login/full-page", wait_until="domcontentloaded")
        print("✅ Navigated to Fidelity login page")
        await page.fill("#dom-username-input", username)
        print("✅ Filled in username")
        await asyncio.sleep(1)
        await page.fill("#dom-pswd-input", password)
        await asyncio.sleep(1)
        print("✅ Filled in password")
        # wait for 10 seconds
        await asyncio.sleep(1)
        await page.click("button:has-text('Log in')")
        print("✅ Clicked on Log in button")
        await page.wait_for_load_state("domcontentloaded")
        print("✅ Waited for load state to be domcontentloaded")
        
        print("✅ Successfully logged into Fidelity!")
        
        # Click on the Positions tab
        try:
            await page.wait_for_selector("#portsum-tab-positions > a", timeout=10000)
            await page.click("#portsum-tab-positions > a")
            print("✅ Clicked on Positions tab")
            await asyncio.sleep(6)  # Wait for the positions page to load
        except Exception as e:
            print(f"⚠️ Could not click on Positions tab: {e}")
        
        # Show mouse coordinates continuously
        print("\n🖱️  Mouse position tracking started. Move your mouse to see coordinates...")
        print("   (Press Ctrl+C to stop)")
        

        await page.mouse.click(1204, 362)
        print("✅ Clicked on 3 dots")
        await page.mouse.click(1128, 390)
        print("✅ Clicked on location 1204 362")
        await asyncio.sleep(2)
        # Inject JavaScript to track mouse position
        await page.evaluate("""
            window._mouseX = 0;
            window._mouseY = 0;
            document.addEventListener('mousemove', function(e) {
                window._mouseX = e.clientX;
                window._mouseY = e.clientY;
            });
        """)

        # Continuously print mouse position
        try:
            while True:
                coords = await page.evaluate("""
                    () => {
                        return {x: window._mouseX, y: window._mouseY};
                    }
                """)
                if coords['x'] > 0 or coords['y'] > 0:  # Only print if mouse has moved
                    print(f"Mouse position: ({coords['x']:.0f}, {coords['y']:.0f})", end='\r')
                await asyncio.sleep(0.1)  # Update 10 times per second
        except KeyboardInterrupt:
            print("\n✅ Mouse tracking stopped")
        print("pause")
        # print the content of the page
        await page.pause()

# Run the script
if __name__ == "__main__":
    asyncio.run(login_to_fidelity(os.getenv("fidelity_user"), os.getenv("fidelity_password")))
