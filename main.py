import os
import re
import csv
import json
import math
import time
import logging
import urllib.parse
import httpx
from bs4 import BeautifulSoup
from typing import List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from openai import OpenAI
import random # <--- Добавьте этот импорт в самое начало файла, где все импорты
# ----------------- НАЛАШТУВАННЯ -----------------
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=GITHUB_TOKEN
)

# ----------------- МОДЕЛІ ДАНИХ (Pydantic) -----------------
class ParsedProduct(BaseModel):
    name: str = Field(description="Нормалізована назва продукту")
    quantity: float = Field(description="Кількість")
    unit: str = Field(description="Одиниця виміру (шт, гр, кг)")
    traits: Optional[str] = Field(None, description="Характеристики")

class VarusItem(BaseModel):
    title: str
    price: str
    url: str
    pack_weight_g: float = Field(1000.0, description="Вага упаковки в грамах")
    is_piece: bool = Field(False, description="Чи продається поштучно")

class MatchedProduct(BaseModel):
    original_request: str
    selected_title: str
    calculated_qty: int
    url: str

# ----------------- ЛОГІКА ПРОГРАМИ -----------------

def parse_input_file(filepath: str) -> List[str]:
    if not os.path.exists(filepath):
        logger.error(f"Файл {filepath} не знайдено!")
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def normalize_with_ai(raw_line: str) -> Optional[ParsedProduct]:
    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Ти експерт з аналізу списків покупок. Витягни дані з рядка. Переведи одиниці виключно в 'шт', 'гр', або 'кг'."},
                {"role": "user", "content": raw_line}
            ],
            response_format=ParsedProduct,
        )
        return completion.choices[0].message.parsed
    except Exception as e:
        logger.error(f"Помилка AI нормалізації: {e}")
        return None

def search_web_for_links(query: str) -> List[str]:
    """Залізобетонний пошук через SerpApi (обходить всі капчі)."""
    serpapi_key = os.getenv("SERPAPI_KEY")
    url = "https://serpapi.com/search"
    params = {
        "engine": "google",
        "q": f"site:varus.ua {query}",
        "api_key": serpapi_key,
        "hl": "uk",
        "num": 3
    }

    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(url, params=params)
            if resp.status_code == 200:
                links = []
                for result in resp.json().get("organic_results", []):
                    href = result.get("link", "")
                    if href.startswith('https://varus.ua') and '/search' not in href and '/ru/' not in href:
                        links.append(href)
                return links
    except Exception as e:
        logger.error(f"Помилка SerpApi: {e}")
    return []

def parse_product_page(url: str) -> Optional[VarusItem]:
    """Парсимо сторінку конкретного товару (вона має SSR і JSON-LD!)."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    }
    try:
        response = httpx.get(url, headers=headers, timeout=15.0, follow_redirects=True)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        title = ""
        price = "0"
        
        # 1. Ідеальний варіант: беремо ціну і назву з JSON-LD (SEO розмітка)
        json_ld_tags = soup.find_all('script', type='application/ld+json')
        for tag in json_ld_tags:
            try:
                data = json.loads(tag.string)
                if isinstance(data, dict) and data.get('@type') == 'Product':
                    title = data.get('name', '')
                    offers = data.get('offers', {})
                    price = str(offers.get('price', 0))
                    break
            except:
                pass
        
        # Якщо JSON-LD не спрацював, беремо з HTML
        if not title:
            title_tag = soup.find('h1', class_='sf-heading__title')
            if title_tag: title = title_tag.text.strip()
            
        if price == "0" or price == "None":
            price_tag = soup.find(class_='sf-price__regular') or soup.find(class_='sf-price__special')
            if price_tag: price = price_tag.text.replace('₴', '').strip()

        if not title: return None
        
        # 2. Витягуємо вагу з div class="count"
        qty_tag = soup.find('div', class_='count')
        qty_text = qty_tag.text.strip().lower() if qty_tag else ""
        
        is_piece = 'шт' in qty_text or 'шт' in title.lower() or 'яйц' in title.lower()
        pack_weight_g = 1000.0
        
        if 'кг' in qty_text and '1 кг' in qty_text:
            pack_weight_g = 1000.0
        else:
            match_g = re.search(r'(\d+)\s*г', qty_text)
            match_kg_text = re.search(r'(\d+(?:\.\d+)?)\s*кг', qty_text)
            if match_g:
                pack_weight_g = float(match_g.group(1))
            elif match_kg_text:
                pack_weight_g = float(match_kg_text.group(1)) * 1000
            else:
                # Шукаємо вагу в назві
                match_title_g = re.search(r'(\d+)\s*г', title.lower())
                match_title_kg = re.search(r'(\d+(?:\.\d+)?)\s*кг', title.lower())
                if match_title_g:
                    pack_weight_g = float(match_title_g.group(1))
                elif match_title_kg:
                    pack_weight_g = float(match_title_kg.group(1)) * 1000
        
        return VarusItem(
            title=title,
            price=price,
            url=url,
            is_piece=is_piece,
            pack_weight_g=pack_weight_g
        )
    except Exception as e:
        logger.error(f"Помилка парсингу {url}: {e}")
        return None

def select_best_match_with_ai(parsed_req: ParsedProduct, found_items: List[VarusItem]) -> Optional[VarusItem]:
    if not found_items: return None
        
    items_text = "\n".join([f"{i}. {item.title}" for i, item in enumerate(found_items)])
    prompt = f"""
    Користувач шукає: {parsed_req.name} {parsed_req.traits if parsed_req.traits else ''}.
    Знайдені товари:
    {items_text}
    
    Поверни ТІЛЬКИ порядковий номер (0, 1, 2...) найкращого збігу. Якщо нічого не підходить, поверни -1.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        idx = int(response.choices[0].message.content.strip())
        if 0 <= idx < len(found_items):
            return found_items[idx]
    except Exception as e:
        pass
    return None

def calculate_optimal_quantity(needed: ParsedProduct, matched_item: VarusItem) -> int:
    if needed.unit == 'шт' or (matched_item.is_piece and needed.unit not in ['гр', 'кг']):
        return math.ceil(needed.quantity)
    
    needed_g = needed.quantity * 1000 if needed.unit == 'кг' else needed.quantity
    return math.ceil(needed_g / matched_item.pack_weight_g)

def export_to_csv(data: List[MatchedProduct], filename: str = "results.csv"):
    try:
        with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(["Продукт", "Товар", "Кількість", "Посилання"])
            for item in data:
                writer.writerow([
                    item.original_request,
                    item.selected_title,
                    item.calculated_qty,
                    item.url
                ])
        logger.info(f"Файл {filename} успішно створено!")
    except Exception as e:
        logger.error(f"Помилка запису CSV: {e}")

# ----------------- ГОЛОВНИЙ ЦИКЛ -----------------

def main():
    logger.info("Запуск програми...")
    if not GITHUB_TOKEN:
        logger.error("Ключ GITHUB_TOKEN не знайдено!")
        return

    raw_lines = parse_input_file("products.txt")
    results: List[MatchedProduct] = []

    for line in raw_lines:
        logger.info(f"Обробка: {line}")
        
        parsed = normalize_with_ai(line)
        if not parsed: continue
            
        logger.info(f"  Нормалізовано: {parsed.name} ({parsed.quantity} {parsed.unit})")
        
        # 1. Шукаємо прямі посилання на товари через веб-пошук
        found_links = search_web_for_links(parsed.name)
        if not found_links:
            logger.warning(f"  Товар '{parsed.name}' не знайдено через веб-пошук.")
            time.sleep(1)
            continue
            
        # 2. Заходимо на кожну сторінку товару і парсимо SSR HTML
        found_items = []
        for link in found_links:
            item = parse_product_page(link)
            if item:
                found_items.append(item)
                
        if not found_items:
            logger.warning(f"  Не вдалося розпарсити сторінки для '{parsed.name}'.")
            continue
            
        # 3. AI вибирає найкращий варіант
        best_match = select_best_match_with_ai(parsed, found_items)
        if not best_match:
            logger.warning(f"  AI не зміг підібрати товар для '{parsed.name}'.")
            time.sleep(1)
            continue
            
        logger.info(f"  Обрано товар: {best_match.title}")
        optimal_qty = calculate_optimal_quantity(parsed, best_match)
        
        results.append(MatchedProduct(
            original_request=line,
            selected_title=best_match.title,
            calculated_qty=optimal_qty,
            url=best_match.url
        ))
        time.sleep(1)

    if results:
        export_to_csv(results)
    else:
        logger.warning("Немає даних для запису в CSV.")
        
    logger.info("Роботу завершено.")

if __name__ == "__main__":
    main()
