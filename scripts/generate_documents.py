"""Generate a synthetic set of scanned freight documents (OCR text) for Brightline Logistics.

Eight document types a driver photographs in the cab. To keep the task from being
trivially solved by the title line, each document:
  * loses its header ~60% of the time (skewed / cut-off scan),
  * is a partial photo ~60% of the time (only 35-70% of the lines survive),
  * shares boilerplate (carrier banner, PRO numbers, addresses, dates) with other types,
  * has its fields shuffled and randomly dropped,
  * gets character-level OCR noise at a per-document rate between 0% and 15%.

Train/val use one pool of company names and cities; test uses a disjoint pool, so
test accuracy measures generalization to unseen customers rather than memorization.

Usage:
    python scripts/generate_documents.py --train 8000 --val 1000 --test 1600
"""
from __future__ import annotations

import argparse
import random
import string
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RAW_DIR, SEED, write_jsonl  # noqa: E402

DOC_TYPES = [
    "bill_of_lading",
    "hazmat_shipping_paper",
    "delivery_receipt",
    "osd_exception_report",
    "freight_invoice",
    "fuel_receipt",
    "scale_ticket",
    "lumper_receipt",
]

# --- Name pools: (train/val pool, test pool) are disjoint ------------------------------
COMPANY_PREFIX = (
    ["Great Plains", "Cedar Ridge", "Ironwood", "Summit", "Blue River", "Prairie", "Northstar",
     "Redstone", "Lakeshore", "Silverline", "Oak Valley", "Granite", "Riverbend", "Pioneer",
     "Heartland", "Maple Leaf", "Frontier", "Golden Field", "Westgate", "Clearwater"],
    ["Copper Canyon", "Harbor Point", "Timberline", "Sandhill", "Bluebonnet", "Eastfield",
     "Stonebridge", "Willow Creek", "High Desert", "Twin Rivers", "Bay Ridge", "Elkhorn"],
)
COMPANY_SUFFIX = (
    ["Foods", "Supply Co", "Manufacturing", "Distribution", "Paper Products", "Chemical",
     "Building Materials", "Beverage", "Packaging", "Farms", "Steel", "Plastics"],
    ["Industries", "Wholesale", "Logistics Center", "Mills", "Grocers", "Components",
     "Fabrication", "Provisions"],
)
CITIES = (
    [("Omaha", "NE"), ("Des Moines", "IA"), ("Kansas City", "MO"), ("Wichita", "KS"),
     ("Lincoln", "NE"), ("Sioux Falls", "SD"), ("Topeka", "KS"), ("Cedar Rapids", "IA"),
     ("Springfield", "MO"), ("Fargo", "ND"), ("Minneapolis", "MN"), ("Denver", "CO")],
    [("Tulsa", "OK"), ("Little Rock", "AR"), ("Amarillo", "TX"), ("Rapid City", "SD"),
     ("Joplin", "MO"), ("Grand Island", "NE"), ("Duluth", "MN"), ("Pueblo", "CO")],
)
STREETS = ["Industrial Pkwy", "Commerce Dr", "W 4th St", "Distribution Way", "Hwy 30",
           "Elm Ave", "Rail Spur Rd", "Logistics Blvd", "N Main St", "Freight Ln"]
TRUCK_STOPS = (
    ["Prairie Fuel Plaza", "Crossroads Travel Center", "Interstate Fuel Stop", "Big Sky Truck Stop",
     "Junction Fuel Depot"],
    ["Mesa Travel Plaza", "Rolling Hills Fuel", "Gateway Truck Center"],
)
COMMODITIES = ["canned vegetables", "corrugated boxes", "PVC pipe fittings", "bagged fertilizer",
               "frozen poultry", "bottled water", "steel coils", "paper towels", "pet food",
               "auto parts", "lumber", "plastic resin pellets", "dry pasta", "furniture"]
HAZMAT = [("UN1203", "Gasoline", "3", "II"), ("UN1993", "Flammable liquid, n.o.s.", "3", "III"),
          ("UN1830", "Sulfuric acid", "8", "II"), ("UN1075", "Petroleum gases, liquefied", "2.1", ""),
          ("UN2794", "Batteries, wet, filled with acid", "8", ""), ("UN1263", "Paint", "3", "II"),
          ("UN3082", "Environmentally hazardous substance, liquid, n.o.s.", "9", "III"),
          ("UN1789", "Hydrochloric acid", "8", "II")]
DAMAGE = ["crushed corner on pallet", "shrink wrap torn, cartons exposed", "wet cartons, water damage",
          "forklift puncture", "load shifted in transit", "broken banding", "cases missing from top tier",
          "leaking drum"]
PERSON_FIRST = ["J.", "M.", "R.", "T.", "K.", "A.", "D.", "L.", "S.", "C."]
PERSON_LAST = ["Nguyen", "Garcia", "Olson", "Patel", "Johnson", "Schmidt", "Brown", "Reyes",
               "Larson", "Kim", "Miller", "Hansen"]

CARRIER_BANNER = ["BRIGHTLINE LOGISTICS", "Brightline Logistics LLC", "BRIGHTLINE LOGISTICS - MC 884213",
                  "Brightline Logistics | USDOT 3120457"]
FOOTERS = ["Page 1 of 1", "Original - Do not copy", "Scanned by driver app v2.3", "Retain for 3 years",
           "Questions? Call dispatch 402-555-0142", ""]


# --- Field helpers ---------------------------------------------------------------------
class Gen:
    def __init__(self, rng: random.Random, pool: int):
        self.r = rng
        self.pool = pool  # 0 = train/val names, 1 = test names

    def company(self):
        return f"{self.r.choice(COMPANY_PREFIX[self.pool])} {self.r.choice(COMPANY_SUFFIX[self.pool])}"

    def address(self):
        city, st = self.r.choice(CITIES[self.pool])
        return f"{self.r.randint(100, 9999)} {self.r.choice(STREETS)}, {city}, {st} {self.r.randint(50000, 69999)}"

    def date(self):
        m, d = self.r.randint(1, 12), self.r.randint(1, 28)
        return self.r.choice([f"{m:02d}/{d:02d}/2026", f"2026-{m:02d}-{d:02d}", f"{m}/{d}/26"])

    def time(self):
        return f"{self.r.randint(0, 23):02d}:{self.r.choice([0, 15, 30, 45]):02d}"

    def pro(self):
        return f"{self.r.randint(10, 99)}-{self.r.randint(1000000, 9999999)}"

    def num(self, k=6):
        return "".join(self.r.choices(string.digits, k=k))

    def person(self):
        return f"{self.r.choice(PERSON_FIRST)} {self.r.choice(PERSON_LAST)}"

    def weight(self, lo=400, hi=42000):
        return f"{self.r.randint(lo, hi):,} lbs"

    def money(self, lo, hi):
        return f"${self.r.uniform(lo, hi):,.2f}"


def _handling_rows(g: Gen, n: int, hazmat: bool = False) -> list[str]:
    rows = []
    for _ in range(n):
        qty = g.r.randint(1, 26)
        unit = g.r.choice(["PLT", "SKID", "CTN", "DRUM", "TOTE"])
        if hazmat:
            un, name, cls, pg = g.r.choice(HAZMAT)
            pg_txt = f", PG {pg}" if pg else ""
            rows.append(f"X  {qty} {unit}  {un}, {name}, {cls}{pg_txt}  {g.weight(200, 9000)}")
        else:
            rows.append(f"{qty} {unit}  {g.r.choice(COMMODITIES)}  NMFC {g.num(6)}  "
                        f"Class {g.r.choice(['50', '55', '65', '70', '85', '100', '125'])}  {g.weight(200, 9000)}")
    return rows


# --- One builder per document type: returns (header_lines, body_fields) ------------------
def bill_of_lading(g: Gen):
    header = [g.r.choice(["STRAIGHT BILL OF LADING", "BILL OF LADING - SHORT FORM", "UNIFORM STRAIGHT BILL OF LADING",
                          "BOL - NOT NEGOTIABLE"])]
    body = [
        f"BOL #: {g.num(8)}", f"Shipper: {g.company()}, {g.address()}",
        f"Consignee: {g.company()}, {g.address()}", f"Bill To (Third Party): {g.company()}",
        f"PRO: {g.pro()}", f"SCAC: BRLG", f"Trailer: {g.num(5)}", f"Seal: {g.num(7)}",
        f"Ship Date: {g.date()}", f"Freight Terms: {g.r.choice(['Prepaid', 'Collect', '3rd Party'])}",
        *_handling_rows(g, g.r.randint(1, 4)),
        "Received, subject to the classifications and tariffs in effect on the date of issue",
        f"Shipper Signature: {g.person()}", f"Carrier Signature: {g.person()}",
    ]
    return header, body


def hazmat_shipping_paper(g: Gen):
    header = [g.r.choice(["HAZARDOUS MATERIALS SHIPPING PAPER", "BILL OF LADING - HAZARDOUS MATERIALS",
                          "STRAIGHT BILL OF LADING (HM)", "DANGEROUS GOODS DECLARATION"])]
    body = [
        f"BOL #: {g.num(8)}", f"Shipper: {g.company()}, {g.address()}",
        f"Consignee: {g.company()}, {g.address()}", f"PRO: {g.pro()}", f"Ship Date: {g.date()}",
        f"Trailer: {g.num(5)}", f"Placards: {g.r.choice(['FLAMMABLE', 'CORROSIVE', 'CLASS 9', 'FLAMMABLE GAS'])}",
        *_handling_rows(g, g.r.randint(1, 3), hazmat=True),
        f"EMERGENCY RESPONSE PHONE: 1-800-{g.num(3)}-{g.num(4)} (24 hr)",
        f"ERG Guide No. {g.r.randint(115, 160)}",
        "This is to certify that the above-named materials are properly classified, packaged, marked and labeled",
        f"Shipper Signature: {g.person()}",
    ]
    return header, body


def delivery_receipt(g: Gen):
    header = [g.r.choice(["DELIVERY RECEIPT", "PROOF OF DELIVERY", "POD - DELIVERY CONFIRMATION",
                          "CONSIGNEE DELIVERY RECEIPT"])]
    pcs = g.r.randint(1, 30)
    body = [
        f"PRO: {g.pro()}", f"Consignee: {g.company()}, {g.address()}", f"Shipper: {g.company()}",
        f"Pieces Shipped: {pcs}", f"Pieces Received: {pcs}", f"Weight: {g.weight()}",
        f"Delivered: {g.date()} {g.time()}", f"Exceptions: {g.r.choice(['None', 'NONE', 'n/a', 'none noted'])}",
        "Received in apparent good order and condition",
        f"Received By (print): {g.person()}", "Consignee Signature: ________",
        f"Driver: {g.person()}  Unit {g.num(4)}",
    ]
    return header, body


def osd_exception_report(g: Gen):
    header = [g.r.choice(["OVER, SHORT & DAMAGED REPORT", "OS&D REPORT", "FREIGHT EXCEPTION REPORT",
                          "DAMAGE / SHORTAGE NOTICE"])]
    shipped = g.r.randint(4, 30)
    short = g.r.randint(1, 3)
    body = [
        f"PRO: {g.pro()}", f"Consignee: {g.company()}, {g.address()}", f"Shipper: {g.company()}",
        f"Pieces Shipped: {shipped}", f"Pieces Received: {shipped - short}",
        f"Exception Type: {g.r.choice(['SHORT', 'DAMAGED', 'OVER', 'REFUSED', 'DAMAGED / SHORT'])}",
        f"Description: {g.r.choice(DAMAGE)}", f"Photos Taken: {g.r.choice(['Yes (4)', 'Y', 'Yes', '2 photos'])}",
        f"Date of Exception: {g.date()} {g.time()}", f"Est. Claim Value: {g.money(80, 6000)}",
        f"Reported By: {g.person()}", "Consignee notation on delivery receipt: YES",
    ]
    return header, body


def freight_invoice(g: Gen):
    header = [g.r.choice(["FREIGHT INVOICE", "INVOICE", "CARRIER INVOICE", "FREIGHT BILL"])]
    body = [
        f"Invoice #: INV-{g.num(7)}", f"Invoice Date: {g.date()}", f"PRO: {g.pro()}",
        f"Bill To: {g.company()}, {g.address()}", f"Origin: {g.company()}", f"Destination: {g.company()}",
        f"Line Haul: {g.money(300, 4200)}", f"Fuel Surcharge: {g.money(40, 900)}",
        f"Accessorial - {g.r.choice(['Liftgate', 'Detention', 'Inside Delivery', 'Residential'])}: {g.money(25, 250)}",
        f"TOTAL DUE: {g.money(400, 5500)}", f"Terms: {g.r.choice(['Net 30', 'Net 15', 'Due on receipt'])}",
        f"Remit To: Brightline Logistics, PO Box {g.num(4)}, Omaha, NE",
    ]
    return header, body


def fuel_receipt(g: Gen):
    stop = g.r.choice(TRUCK_STOPS[g.pool])
    city, st = g.r.choice(CITIES[g.pool])
    gal = g.r.uniform(40, 180)
    ppg = g.r.uniform(3.2, 4.6)
    header = [g.r.choice([stop.upper(), f"{stop} #{g.r.randint(10, 400)}", "FUEL RECEIPT", "CUSTOMER COPY"])]
    body = [
        f"{city}, {st}", f"Date: {g.date()} Time: {g.time()}", f"Pump: {g.r.randint(1, 24)}",
        f"Product: {g.r.choice(['DIESEL #2', 'ULSD', 'DIESEL'])}", f"Gallons: {gal:.3f}",
        f"Price/Gal: ${ppg:.3f}", f"DEF Gallons: {g.r.uniform(0, 12):.2f}", f"Fuel Total: ${gal * ppg:,.2f}",
        f"Unit #: {g.num(4)}", f"Odometer: {g.r.randint(100000, 900000)}",
        f"Payment: {g.r.choice(['FLEET CARD ****', 'COMDATA ****', 'EFS ****'])}{g.num(4)}",
        f"Auth: {g.num(6)}", "THANK YOU - DRIVE SAFE",
    ]
    return header, body


def scale_ticket(g: Gen):
    steer = g.r.randint(10500, 12500)
    drive = g.r.randint(26000, 34000)
    trailer = g.r.randint(24000, 34000)
    header = [g.r.choice(["CERTIFIED WEIGHT TICKET", "SCALE TICKET", "CERTIFIED SCALE - WEIGH TICKET",
                          "TRUCK SCALE RECEIPT"])]
    body = [
        f"Ticket #: {g.num(7)}", f"Location: {g.r.choice(TRUCK_STOPS[g.pool])}", f"Date: {g.date()} {g.time()}",
        f"Steer Axle: {steer:,} lb", f"Drive Axle: {drive:,} lb", f"Trailer Axle: {trailer:,} lb",
        f"GROSS WEIGHT: {steer + drive + trailer:,} lb", f"Truck #: {g.num(4)}  Trailer #: {g.num(5)}",
        f"Weighmaster: {g.person()}", f"Reweigh: {g.r.choice(['No', 'Yes - $3.00', 'N'])}",
        "Weight guaranteed or we pay the fine",
    ]
    return header, body


def lumper_receipt(g: Gen):
    header = [g.r.choice(["LUMPER RECEIPT", "UNLOADING SERVICE RECEIPT", "LUMPER SERVICES", "WAREHOUSE UNLOAD RECEIPT"])]
    body = [
        f"Receipt #: {g.num(6)}", f"Warehouse: {g.company()}, {g.address()}", f"Date: {g.date()}",
        f"Door: {g.r.randint(1, 60)}", f"PO #: {g.num(8)}", f"Carrier: Brightline Logistics  PRO {g.pro()}",
        f"Pallets Unloaded: {g.r.randint(4, 30)}", f"Restack/Break Down: {g.r.choice(['Yes', 'No'])}",
        f"Lumper Fee: {g.money(75, 450)}", f"Paid By: {g.r.choice(['Comchek', 'Cash', 'EFS Check', 'Company Card'])}",
        f"Unloader: {g.person()}",
    ]
    return header, body


BUILDERS = {name: globals()[name] for name in DOC_TYPES}


# --- OCR noise -------------------------------------------------------------------------
OCR_CONFUSIONS = {
    "O": "0", "0": "O", "l": "1", "1": "l", "I": "1", "S": "5", "5": "S", "B": "8", "8": "B",
    "e": "c", "c": "e", "a": "o", "n": "r", "G": "6", "Z": "2", ".": ",", ",": ".",
}


def ocr_noise(text: str, rate: float, r: random.Random) -> str:
    out = []
    for ch in text:
        if ch == "\n" or r.random() >= rate:
            out.append(ch)
            continue
        op = r.random()
        if op < 0.5 and ch in OCR_CONFUSIONS:
            out.append(OCR_CONFUSIONS[ch])
        elif op < 0.7:
            continue  # dropped character
        elif op < 0.85:
            out.append(ch + r.choice(string.ascii_letters + " .'-"))  # spurious character
        elif ch == " ":
            continue  # merged words
        else:
            out.append(ch.swapcase())
    return "".join(out)


def make_document(doc_type: str, r: random.Random, pool: int) -> dict:
    g = Gen(r, pool)
    header, body = BUILDERS[doc_type](g)

    keep_header = r.random() > 0.6
    body = [line for line in body if r.random() > 0.25]  # randomly missing fields
    # Keep the first two lines (identifiers) roughly in place, shuffle the rest locally.
    head, tail = body[:2], body[2:]
    for i in range(len(tail) - 1):
        if r.random() < 0.3:
            tail[i], tail[i + 1] = tail[i + 1], tail[i]
    body = head + tail
    # Partial scan: the driver photographed only part of the page.
    cropped = r.random() < 0.6
    if cropped:
        keep = max(2, round(len(body) * r.uniform(0.35, 0.7)))
        start = r.randint(0, len(body) - keep)
        body = body[start:start + keep]
        keep_header = keep_header and start == 0

    lines = []
    if r.random() < 0.7:
        lines.append(r.choice(CARRIER_BANNER))
    if keep_header:
        lines.extend(header)
    lines.extend(body)
    footer = r.choice(FOOTERS)
    if footer:
        lines.append(footer)

    noise_rate = r.uniform(0.0, 0.15)
    text = ocr_noise("\n".join(lines), noise_rate, r)
    return {
        "text": text,
        "label": f"doc:{doc_type}",
        "source": "synthetic_docs",
        "meta": {"header_visible": keep_header, "cropped": cropped, "ocr_noise": round(noise_rate, 4), "name_pool": pool},
    }


def generate_split(n: int, seed: int, pool: int) -> list[dict]:
    r = random.Random(seed)
    rows = [make_document(DOC_TYPES[i % len(DOC_TYPES)], r, pool) for i in range(n)]
    r.shuffle(rows)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=int, default=8000)
    ap.add_argument("--val", type=int, default=1000)
    ap.add_argument("--test", type=int, default=1600)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    out = RAW_DIR / "synthetic_docs"
    splits = {
        "train": generate_split(args.train, args.seed, pool=0),
        "val": generate_split(args.val, args.seed + 1, pool=0),
        "test": generate_split(args.test, args.seed + 2, pool=1),  # unseen companies/cities
    }
    for name, rows in splits.items():
        write_jsonl(out / f"{name}.jsonl", rows)
        print(f"{name:5s}: {len(rows):5d} documents -> {out / f'{name}.jsonl'}")
    print("\nExample (test):\n" + "-" * 60 + "\n" + splits["test"][0]["text"] + "\n" + "-" * 60)
    print(f"label: {splits['test'][0]['label']}  meta: {splits['test'][0]['meta']}")


if __name__ == "__main__":
    main()
