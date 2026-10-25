"""Build the sample shop database (deterministic) and its text-to-SQL evaluation set."""
import json
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "sample_db"
OUT.mkdir(parents=True, exist_ok=True)
DB = OUT / "shop.db"
rnd = random.Random(2026)

CITIES = [("Pune", "India"), ("Mumbai", "India"), ("Bengaluru", "India"), ("Delhi", "India"), ("London", "UK"),
          ("Manchester", "UK"), ("Berlin", "Germany"), ("Munich", "Germany"), ("Austin", "USA"), ("Seattle", "USA")]
FIRST = ["Aarav", "Diya", "Kabir", "Meera", "Rohan", "Ananya", "Ishaan", "Priya", "Oliver", "Amelia", "Lukas",
         "Hannah", "Noah", "Emma", "Liam", "Sofia", "Arjun", "Kavya", "Felix", "Zara"]
LAST = ["Sharma", "Iyer", "Reddy", "Menon", "Smith", "Brown", "Muller", "Schmidt", "Johnson", "Garcia", "Patel",
        "Kapoor", "Wilson", "Fischer", "Taylor", "Rao"]
PRODUCTS = [("Wireless Mouse", "Electronics", 899), ("Mechanical Keyboard", "Electronics", 4599),
            ("USB-C Hub", "Electronics", 2499), ("27-inch Monitor", "Electronics", 15999),
            ("Noise Cancelling Headphones", "Electronics", 12999), ("Laptop Stand", "Accessories", 1799),
            ("Desk Lamp", "Accessories", 1299), ("Ergonomic Chair", "Furniture", 18999),
            ("Standing Desk", "Furniture", 32999), ("Bookshelf", "Furniture", 7499),
            ("Notebook Pack", "Stationery", 349), ("Gel Pens (10)", "Stationery", 249),
            ("Coffee Beans 1kg", "Grocery", 1199), ("Green Tea (100 bags)", "Grocery", 599),
            ("Water Bottle", "Accessories", 699), ("Backpack", "Accessories", 2799)]

if DB.exists():
    DB.unlink()
con = sqlite3.connect(DB)
con.executescript("""
CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT NOT NULL, email TEXT, city TEXT, country TEXT,
                        signup_date TEXT);
CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT NOT NULL, category TEXT, price REAL);
CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER REFERENCES customers(id), order_date TEXT,
                     status TEXT);
CREATE TABLE order_items (id INTEGER PRIMARY KEY, order_id INTEGER REFERENCES orders(id),
                          product_id INTEGER REFERENCES products(id), quantity INTEGER, unit_price REAL);
""")
for i in range(1, 121):
    city, country = rnd.choice(CITIES)
    first, last = rnd.choice(FIRST), rnd.choice(LAST)
    d = date(2024, 1, 1) + timedelta(days=rnd.randint(0, 600))
    con.execute("INSERT INTO customers VALUES (?,?,?,?,?,?)",
                (i, f"{first} {last}", f"{first.lower()}.{last.lower()}{i}@example.com", city, country, d.isoformat()))
for i, (n, c, p) in enumerate(PRODUCTS, 1):
    con.execute("INSERT INTO products VALUES (?,?,?,?)", (i, n, c, p))
item_id = 1
for oid in range(1, 601):
    cust = rnd.randint(1, 120)
    d = date(2025, 1, 1) + timedelta(days=rnd.randint(0, 540))
    status = rnd.choices(["delivered", "shipped", "cancelled", "returned"], [75, 12, 8, 5])[0]
    con.execute("INSERT INTO orders VALUES (?,?,?,?)", (oid, cust, d.isoformat(), status))
    for _ in range(rnd.randint(1, 4)):
        pid = rnd.randint(1, len(PRODUCTS))
        price = PRODUCTS[pid - 1][2] * rnd.choice([1, 1, 1, 0.9, 0.85])
        con.execute("INSERT INTO order_items VALUES (?,?,?,?,?)", (item_id, oid, pid, rnd.randint(1, 3), round(price, 2)))
        item_id += 1
con.commit()
con.close()

EVAL = [
    ("How many customers are there?", "SELECT count(*) FROM customers"),
    ("How many customers live in India?", "SELECT count(*) FROM customers WHERE country = 'India'"),
    ("Which product is the most expensive?", "SELECT name FROM products ORDER BY price DESC LIMIT 1"),
    ("List the product categories and how many products each has.",
     "SELECT category, count(*) FROM products GROUP BY category"),
    ("How many orders were cancelled?", "SELECT count(*) FROM orders WHERE status = 'cancelled'"),
    ("What is the total revenue from delivered orders?",
     "SELECT round(sum(oi.quantity * oi.unit_price), 2) FROM order_items oi JOIN orders o ON o.id = oi.order_id "
     "WHERE o.status = 'delivered'"),
    ("Which city has the most customers?", "SELECT city FROM customers GROUP BY city ORDER BY count(*) DESC LIMIT 1"),
    ("What are the top 3 products by units sold?",
     "SELECT p.name FROM order_items oi JOIN products p ON p.id = oi.product_id GROUP BY p.id "
     "ORDER BY sum(oi.quantity) DESC LIMIT 3"),
    ("How many orders were placed in 2025?", "SELECT count(*) FROM orders WHERE strftime('%Y', order_date) = '2025'"),
    ("What is the average order value for delivered orders?",
     "SELECT round(avg(total), 2) FROM (SELECT sum(oi.quantity * oi.unit_price) AS total FROM order_items oi "
     "JOIN orders o ON o.id = oi.order_id WHERE o.status = 'delivered' GROUP BY o.id)"),
    ("Which customer spent the most in total?",
     "SELECT c.name FROM customers c JOIN orders o ON o.customer_id = c.id JOIN order_items oi ON oi.order_id = o.id "
     "GROUP BY c.id ORDER BY sum(oi.quantity * oi.unit_price) DESC LIMIT 1"),
    ("Revenue by product category, highest first.",
     "SELECT p.category, round(sum(oi.quantity * oi.unit_price), 2) AS revenue FROM order_items oi "
     "JOIN products p ON p.id = oi.product_id GROUP BY p.category ORDER BY revenue DESC"),
    ("How many customers have never placed an order?",
     "SELECT count(*) FROM customers WHERE id NOT IN (SELECT DISTINCT customer_id FROM orders)"),
    ("How many orders were returned in each country?",
     "SELECT c.country, count(*) FROM orders o JOIN customers c ON c.id = o.customer_id WHERE o.status = 'returned' "
     "GROUP BY c.country"),
    ("Which month of 2025 had the most orders?",
     "SELECT strftime('%m', order_date) FROM orders WHERE order_date LIKE '2025%' GROUP BY 1 ORDER BY count(*) DESC LIMIT 1"),
]
(OUT / "eval.jsonl").write_text("\n".join(json.dumps({"question": q, "gold_sql": s}) for q, s in EVAL) + "\n",
                                encoding="utf-8")
print("built", DB, DB.stat().st_size, "bytes;", len(EVAL), "eval questions")
