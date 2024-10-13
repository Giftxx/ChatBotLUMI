#  Description ด้านบน
# Details รายละเอียดผลิตภัณ
import requests
from bs4 import BeautifulSoup

def scrape_product(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    name = soup.select_one('#site-content > div.block.block-system.block-system-main > div > div > div > div.product-full.product-full-v1.js-product.js-product-v1 > div.product-full__container > div.product-full__content.top > div > h1')
    description = soup.select_one('#site-content > div.block.block-system.block-system-main > div > div > div > div.product-full.product-full-v1.js-product.js-product-v1 > div.product-full__container > div.product-full__content.bottom > div > div.product-full__description-wrapper.js-product-full__description-wrapper > div')
    details = soup.select_one('div.product-full__description.product-full__accordion div.product-full__accordion__panel')
    use = soup.select_one('#site-content div:nth-child(2) div:nth-child(5) div:nth-child(5) div:nth-child(3) div:nth-child(2) p:nth-child(1)')
    return {
        'URL': url,
        'Name': name.get_text(strip=True) if name else 'Not found',
        'Description': description.get_text(strip=True) if description else 'Not found',
        'Details': details.get_text(strip=True) if details else 'Not found',
        'use': use.get_text(strip=True) if use else 'Not found'
    }

urls = [
    'https://www.lamer.co.th/product/5834/130879/face/moisturizers/the-new-rejuvenating-night-cream#/sku/190531',
    'https://www.lamer.co.th/product/22153/132443/sets/the-creme-de-la-mer-duet--creme-de-la-mer-60-ml-creme-de-la-mer-15-ml',
    'https://www.lamer.co.th/product/22153/132445/sets/the-calming-hydration-collection--creme-de-la-mer-60-ml-the-eye-concentrate-15-ml',
    'https://www.lamer.co.th/product/22153/132446/sets/the-moisturizing-soft-cream-duet--the-moisturizing-soft-cream-60-ml-the-moisturizing-soft-cream-15-ml',
    'https://www.lamer.co.th/product/5834/124395/face/moisturizers/the-new-moisturizing-fresh-cream#/sku/181919',
    'https://www.lamer.co.th/product/5834/12343/face/moisturizers/creme-de-la-mer-#/sku/60983',
    'https://www.lamer.co.th/product/5834/48607/face/moisturizers/the-moisturizing-matte-lotion/moisturizer-for-oily-skin#/sku/80451',
    'https://www.lamer.co.th/product/5834/42790/face/moisturizers/the-moisturizing-soft-lotion/lotion-for-dry-skin#/sku/73000',
    'https://www.lamer.co.th/product/5834/104371/face/moisturizers/the-moisturizing-soft-cream#/sku/153757',
    'https://www.lamer.co.th/product/11484/88916/prep/the-hydrating-infused-emulsion/lightweight-moisturizer-for-face#/sku/133945',
    'https://www.lamer.co.th/product/20658/78410/collections/genaissance-de-la-mertm/genaissance-de-la-mer-the-concentrated-night-balm/night-cream#/sku/120025'
]

results = []
for url in urls:
    product_info = scrape_product(url)
    results.append(product_info)
    print(f"Scraped: {product_info['Name']}")

# แสดงผลลัพธ์ทั้งหมดโดยไม่ตัดข้อความ
for result in results:
    print(f"\nProduct: {result['Name']}")
    print(f"URL: {result['URL']}")
    print(f"Description: {result['Description']}")
    print(f"Details: {result['Details']}"),
    print(f"use: {result['use']}")
    print("-" * 50)  # เส้นแบ่งระหว่างผลิตภัณฑ์