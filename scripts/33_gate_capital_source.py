"""Step 33: does the global footballdatabase capital beat clubelo+floor?

Same LOTO gate, comparing two capital sources head to head: the deployed
clubelo (Europe-only, floor-imputed) vs footballdatabase (global, 89% cover).
Each gets its own fitted beta per fold; we compare pooled OOS log-loss. The
fdb source removes the Europe bias by construction; this asks whether it also
predicts at least as well.
"""
import json, sys
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
from wc26.data import load_results
from wc26.dixon_coles import DixonColes

TOURN = [("wc2014","FIFA World Cup","2014-06-12","2014-07-13"),("wc2018","FIFA World Cup","2018-06-14","2018-07-15"),
         ("wc2022","FIFA World Cup","2022-11-20","2022-12-18"),("euro2016","UEFA Euro","2016-06-10","2016-07-10"),
         ("euro2020","UEFA Euro","2021-06-11","2021-07-11"),("euro2024","UEFA Euro","2024-06-14","2024-07-14")]
clubelo = pd.read_csv("outputs/capital.csv")
fdb = pd.read_csv("outputs/capital_fdb.csv")
results = load_results(); elo = pd.read_parquet("outputs/elo_history.parquet")
df = results.dropna(subset=["home_score","away_score"]).copy()
key=["date","home_team","away_team"]; df["dup"]=df.groupby(key).cumcount(); elo["dup"]=elo.groupby(key).cumcount()
df = df.merge(elo[key+["dup","elo_home_pre","elo_away_pre"]],on=key+["dup"],validate="1:1").sort_values("date")

def packs(capdf):
    P={}
    for slug,comp,start,end in TOURN:
        fd=pd.Timestamp(start); model=DixonColes().fit(df[df.date>=fd-pd.DateOffset(years=20)],fd)
        test=df[(df.tournament==comp)&(df.date>=start)&(df.date<=end)]
        cz=capdf[capdf.tournament==slug].set_index("team")["capital_z"]
        rows=[]
        for r in test.itertuples(index=False):
            lh,la=model.predict_lambdas(r.elo_home_pre,r.elo_away_pre,neutral=bool(r.neutral))
            d=float(cz.get(r.home_team,0.0)-cz.get(r.away_team,0.0))
            a=0 if r.home_score>r.away_score else (1 if r.home_score==r.away_score else 2)
            rows.append((lh,la,d,a))
        P[slug]=(model,rows)
    return P

def gate(P):
    def llv(slug,b):
        m,rows=P[slug]
        return np.array([-np.log(max(m.outcome_probs(lh*np.exp(b*d),la*np.exp(-b*d))[a],1e-12)) for lh,la,d,a in rows])
    pool0=pool1=n=0.0
    for held,*_ in TOURN:
        others=[s for s,*_ in TOURN if s!=held]
        b=minimize_scalar(lambda b: sum(float(llv(s,b).sum()) for s in others),bounds=(-.5,.5),method="bounded").x
        nn=len(P[held][1]); pool0+=llv(held,0).mean()*nn; pool1+=llv(held,b).mean()*nn; n+=nn
    return pool0/n, pool1/n

for name,capdf in [("clubelo+floor (deployed)",clubelo),("footballdatabase (global)",fdb)]:
    base,cap=gate(packs(capdf))
    print(f"{name:28s}: base {base:.4f} -> +capital {cap:.4f}  ({cap-base:+.5f})")
