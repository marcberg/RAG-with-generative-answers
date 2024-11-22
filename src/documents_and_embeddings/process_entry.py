from src.documents_and_embeddings.keywords import extract_keywords

def process_entry_products_old(entry):
    name = entry.get("name")
    description = entry.get("description")

    name = name if isinstance(name, str) else "No name available"
    description = description if isinstance(description, str) else ""
    
    text = f"product: {name} {description}".strip()

    if not text:
        text = "No content available"
    
    return text

def process_entry_products(entry):
    name = entry.get("name", "null")
    product_type = entry.get("type", "null")
    price = entry.get("price", "null")
    description = entry.get("description", "null")
    if description != "null":
        description_keywords = extract_keywords(description)
        description = ' '.join(description_keywords)

    manufacturer = entry.get("manufacturer", "null")

    # Extract and format categories (limit to top 5)
    categories = entry.get("category", [])
    category_names = " - ".join([cat.get("name", "Unknown category") for cat in categories[:5]])
    if not category_names:
        category_names = "No categories available"
    
    # Create the structured text
    text = (
        "product: "
        f"Name: {name}. "
        f"Type: {product_type}. "
        f"Price: {price}. "
        f"Category: {category_names}. "
        f"Manufacturer: {manufacturer}. "
        f"Description: {description}."
    ).strip()
    
    return text

def process_entry_categories(entry):
    name = entry.get("name")

    name = name if isinstance(name, str) else "No name available"
    
    text = f"category: {name}".strip()

    if not text:
        text = "No content available"
    
    return text


def process_entry_stores(entry):
    name = entry.get("name")
    city = entry.get("city")

    name = name if isinstance(name, str) else "No name available"
    city = city if isinstance(city, str) else ""
    
    text = f"store: {name} {city}".strip()

    if not text:
        text = "No content available"
    
    return text