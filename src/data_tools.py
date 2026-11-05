import json
import unicodedata
from typing import Optional

import pandas as pd

from src.data_loader import load_all

MAX_RESULTS = 8

_data = load_all()


def _stem(word: str) -> str:
    """Lowercase, strip accents and apply a naive singular form (violões -> violao)."""
    word = unicodedata.normalize("NFKD", word.lower())
    word = "".join(ch for ch in word if not unicodedata.combining(ch))
    if word.endswith("oes"):
        word = word[:-3] + "ao"
    elif word.endswith("s") and len(word) > 3:
        word = word[:-1]
    return word


def _normalize(text: str) -> str:
    return " ".join(_stem(w) for w in str(text).split())


def _clean(value):
    """Convert NaN to None so the model receives a clean value."""
    return None if pd.isna(value) else value


def _build_products() -> pd.DataFrame:
    products = _data["products"].merge(
        _data["categories"][["category_id", "name"]].rename(columns={"name": "category"}),
        on="category_id",
        how="left",
    )
    promos = _data["promotions"]
    best_discount = (
        promos[promos["is_active"] == 1]
        .groupby("product_id")["discount_percent"]
        .max()
        .rename("discount_percent")
    )
    products = products.merge(best_discount, on="product_id", how="left")
    products["promo_price_brl"] = (
        products["price_brl"] * (1 - products["discount_percent"] / 100)
    ).round(2)
    products["search_text"] = (
        products["name"] + " " + products["category"].fillna("")
    ).map(_normalize)
    return products


_products = _build_products()


def _summary(row) -> dict:
    has_promo = pd.notna(row["discount_percent"])
    return {
        "product_id": int(row["product_id"]),
        "name": row["name"],
        "category": row["category"],
        "price_brl": float(row["price_brl"]),
        "promo_discount_percent": int(row["discount_percent"]) if has_promo else None,
        "promo_price_brl": float(row["promo_price_brl"]) if has_promo else None,
        "stock_quantity": int(row["stock_quantity"]),
        "in_stock": bool(row["stock_quantity"] > 0),
        "status": row["status"],
    }


def _parse_specs(raw) -> dict:
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return {}


def list_categories() -> dict:
    """List the product categories sold by the store."""
    return {"categories": _data["categories"]["name"].tolist()}


def search_products(
    query: Optional[str] = None,
    category: Optional[str] = None,
    max_price: Optional[float] = None,
    in_stock_only: bool = True,
) -> dict:
    """Search the active product catalog.

    Args:
        query: Only short keywords such as product type, brand or model
            (e.g. "violao yamaha"). Do not include filler words.
        category: Category name, e.g. "Violões".
        max_price: Maximum price in BRL. ALWAYS set this when the customer mentions
            a budget ("até R$ 1000", "menos de 500", "no máximo..."). Promotional
            prices are considered.

    Returns:
        A dict with the total found and up to 8 products, cheapest first.
    """
    df = _products[_products["status"] == "active"]
    if query:
        for token in _normalize(query).split():
            df = df[df["search_text"].str.contains(token, regex=False)]
    if category:
        df = df[df["category"].map(_normalize).str.contains(_normalize(category), regex=False)]
    if max_price is not None:
        effective = df["promo_price_brl"].fillna(df["price_brl"])
        df = df[effective <= max_price]
    if in_stock_only:
        df = df[df["stock_quantity"] > 0]
    df = df.sort_values("price_brl")
    return {
        "filters_applied": {
            "query": query,
            "category": category,
            "max_price": max_price,
            "in_stock_only": in_stock_only,
        },
        "total_found": int(len(df)),
        "products": [_summary(row) for _, row in df.head(MAX_RESULTS).iterrows()],
    }


def get_product_details(product_name: str) -> dict:
    """Get price, promotion, stock, description and specs of a specific product.

    Args:
        product_name: Product name or distinctive part of it, e.g. "Takamine GD20".
    """
    df = _products
    for token in _normalize(product_name).split():
        df = df[df["search_text"].str.contains(token, regex=False)]
    if df.empty:
        return {"error": "Product not found."}
    results = []
    for _, row in df.head(5).iterrows():
        item = _summary(row)
        item["description"] = row["description"]
        item["specs"] = _parse_specs(row["specs"])
        results.append(item)
    return {"total_matches": int(len(df)), "products": results}


def get_active_promotions() -> dict:
    """List the promotions that are currently active, with discounted prices."""
    df = _products[_products["discount_percent"].notna()]
    promos = _data["promotions"]
    promos = promos[promos["is_active"] == 1][["product_id", "description"]]
    df = df.merge(promos, on="product_id", how="left", suffixes=("", "_promo"))
    return {
        "promotions": [
            {**_summary(row), "promotion_name": _clean(row["description_promo"])}
            for _, row in df.iterrows()
        ]
    }


def get_order_status(order_id: int, customer_email: str) -> dict:
    """Get the status and details of an order.

    The customer must provide both the order number and the e-mail used in the
    purchase. Never call this without both.

    Args:
        order_id: Order number.
        customer_email: E-mail of the customer who made the order.
    """
    not_found = {"error": "No order found for this order number and e-mail."}
    orders = _data["orders"]
    order = orders[orders["order_id"] == int(order_id)]
    if order.empty:
        return not_found
    order = order.iloc[0]

    customers = _data["customers"]
    customer = customers[customers["customer_id"] == order["customer_id"]].iloc[0]
    if customer["email"].strip().lower() != customer_email.strip().lower():
        return not_found

    items = _data["order_items"].merge(
        _data["products"][["product_id", "name"]], on="product_id", how="left"
    )
    items = items[items["order_id"] == int(order_id)]
    return {
        "order_id": int(order["order_id"]),
        "order_date": order["order_date"],
        "status": order["status"],
        "total_brl": float(order["total_brl"]),
        "payment_method": order["payment_method"],
        "tracking_code": _clean(order["tracking_code"]),
        "estimated_delivery": _clean(order["estimated_delivery"]),
        "notes": _clean(order["notes"]),
        "items": [
            {"product": row["name"], "quantity": int(row["quantity"])}
            for _, row in items.iterrows()
        ],
    }