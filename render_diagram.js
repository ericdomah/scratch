const puppeteer = require('puppeteer');

(async () => {
  const browser = await puppeteer.launch();
  const page = await browser.newPage();
  await page.setViewport({ width: 1200, height: 1000, deviceScaleFactor: 2 });
  await page.goto('file:///C:/Users/User/Downloads/scratch-main/thesis/images/architecture_diagram.html', { waitUntil: 'networkidle0' });
  const element = await page.$('.mermaid');
  if(element) {
      await element.screenshot({ path: 'C:/Users/User/Downloads/scratch-main/thesis/images/architecture_diagram.png' });
      console.log('Diagram rendered and saved!');
  } else {
      console.log('Mermaid element not found');
  }
  await browser.close();
})();
