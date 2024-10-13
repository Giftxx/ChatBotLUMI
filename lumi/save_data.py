from flask import Flask, request, jsonify
from selenium import webdriver
from selenium.webdriver.common.by import By
import chromedriver_autoinstaller
import time
import csv

# setup chrome options
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--disable-dev-shm-usage')

# automatically install chromedriver
chromedriver_autoinstaller.install()

app = Flask(__name__)
port = "7092"

@app.route('/')
def index():
    return "<h1>Test API</h1>"

@app.route('/api', methods=['GET'])
def api():
    if request.method == 'GET':
        query_msg = request.args.get('msg', '').strip()

        if query_msg == "product_details":
            try:
                # List of product URLs
                product_urls = [
                    "https://www.lamer.co.th/product/5834/130879/face/moisturizers/the-new-rejuvenating-night-cream#/sku/190531",
                    "https://www.lamer.co.th/product/22153/132443/sets/the-creme-de-la-mer-duet--creme-de-la-mer-60-ml-creme-de-la-mer-15-ml",
                    "https://www.lamer.co.th/product/22153/132445/sets/the-calming-hydration-collection--creme-de-la-mer-60-ml-the-eye-concentrate-15-ml",
                    "https://www.lamer.co.th/product/22153/132446/sets/the-moisturizing-soft-cream-duet--the-moisturizing-soft-cream-60-ml-the-moisturizing-soft-cream-15-ml",
                    "https://www.lamer.co.th/product/5834/124395/face/moisturizers/the-new-moisturizing-fresh-cream#/sku/181919",
                    "https://www.lamer.co.th/product/5834/12343/face/moisturizers/creme-de-la-mer-#/sku/60983",
                    "https://www.lamer.co.th/product/5834/48607/face/moisturizers/the-moisturizing-matte-lotion/moisturizer-for-oily-skin#/sku/80451",
                    "https://www.lamer.co.th/product/5834/42790/face/moisturizers/the-moisturizing-soft-lotion/lotion-for-dry-skin#/sku/73000",
                    "https://www.lamer.co.th/product/5834/104371/face/moisturizers/the-moisturizing-soft-cream#/sku/153757",
                    "https://www.lamer.co.th/product/11484/88916/prep/the-hydrating-infused-emulsion/lightweight-moisturizer-for-face#/sku/133945",
                    "https://www.lamer.co.th/product/20658/78410/collections/genaissance-de-la-mertm/genaissance-de-la-mer-the-concentrated-night-balm/night-cream#/sku/120025"
                ]

                result = []
                
                # Open a CSV file to write the output
                with open('product_data.csv', mode='w', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    
                    # Write the header of the CSV file
                    writer.writerow(["Product Name", "Description", "Price", "Details", "Ingredients", "How to Use"])

                    for url in product_urls:
                        driver = webdriver.Chrome(options=chrome_options)
                        driver.get(url)
                        driver.implicitly_wait(5)

                        time.sleep(1)

                        product_info = {}

                        # Get product details based on XPaths and clean the data
                        try:
                            product_name = driver.find_element(By.XPATH, '//*[@id="site-content"]/div[2]/div/div/div/div[1]/div[1]/div[2]/div/h1').text.strip().replace("\n", " ")
                            product_info["ชื่อสินค้า"] = product_name
                        except Exception:
                            product_info["ชื่อสินค้า"] = ""

                        try:
                            description = driver.find_element(By.XPATH, '//*[@id="site-content"]/div[2]/div/div/div/div[1]/div[1]/div[3]/div/div[1]/div').text.strip().replace("\n", " ")
                            product_info["คำอธิบายสินค้า"] = description
                        except Exception:
                            product_info["คำอธิบายสินค้า"] = ""

                        try:
                            price = driver.find_element(By.XPATH, '//*[@id="site-content"]/div[2]/div/div/div/div[1]/div[1]/div[3]/div/div[2]/div[1]/div/span').text.strip().replace("\n", " ")
                            product_info["ราคาสินค้า"] = price
                        except Exception:
                            product_info["ราคาสินค้า"] = ""

                        try:
                            product_details = driver.find_element(By.XPATH, '//*[@id="site-content"]/div[2]/div/div/div/div[1]/div[1]/div[3]/div/div[5]/div[5]/div[1]/div[2]/p').text.strip().replace("\n", " ")
                            product_info["รายระเอียดสินค้า"] = product_details
                        except Exception:
                            product_info["รายระเอียดสินค้า"] = ""

                        try:
                            ingredients = driver.find_element(By.XPATH, '//*[@id="site-content"]/div[2]/div/div/div/div[1]/div[1]/div[3]/div/div[5]/div[5]/div[2]/div[2]/p[1]/a').text.strip().replace("\n", " ")
                            product_info["ส่วนผสมสำคัญ"] = ingredients
                        except Exception:
                            product_info["ส่วนผสมสำคัญ"] = ""

                        try:
                            how_to_use = driver.find_element(By.XPATH, '//*[@id="site-content"]/div[2]/div/div/div/div[1]/div[1]/div[3]/div/div[5]/div[5]/div[3]/div[2]/p').text.strip().replace("\n", " ")
                            product_info["การใช้งาน"] = how_to_use
                        except Exception:
                            product_info["การใช้งาน"] = ""

                        # Write the product info into the CSV file
                        writer.writerow([
                            product_info.get("ชื่อสินค้า", ""),
                            product_info.get("คำอธิบายสินค้า", ""),
                            product_info.get("ราคาสินค้า", ""),
                            product_info.get("รายระเอียดสินค้า", ""),
                            product_info.get("ส่วนผสมสำคัญ", ""),
                            product_info.get("การใช้งาน", "")
                        ])

                        driver.quit()  # Close driver after each product

                # Return the final result
                return jsonify({"message": "Data has been successfully written to CSV file."})

            except Exception as e:
                return jsonify({"error": f"An error occurred: {str(e)}"}), 500

        else:
            return jsonify({"error": "No matching product group found."}), 404

# Start the Flask server 
if __name__ == '__main__':
    app.run(port=port, debug=True)
