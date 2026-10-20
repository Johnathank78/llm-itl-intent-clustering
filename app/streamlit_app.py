"""Explorateur des résultats : découverte d'intentions par LLM-in-the-loop (MASSIVE fr-FR).

Lancer en local, depuis la racine du dépôt :
    pip install -r app/requirements.txt
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import donnees as D

BLEU = "#2a78d6"      # série mise en avant (le tamis LLM, prompt v1)
ORANGE = "#eb6834"    # seconde série (prompt v2)
GRIS = "#b9b7ae"      # baselines, éléments de contexte
ENCRE = "#52514e"
GRILLE = "#e7e6e2"

st.set_page_config(page_title="Intentions par LLM-in-the-loop", page_icon="🧭", layout="wide")


def fr(x: float, d: int = 3) -> str:
    return f"{x:.{d}f}".replace(".", ",")


def style(fig: go.Figure, hauteur: int = 360) -> go.Figure:
    fig.update_layout(
        height=hauteur, margin=dict(l=10, r=20, t=30, b=10), plot_bgcolor="white", paper_bgcolor="white",
        font=dict(color=ENCRE, size=13), barcornerradius=4, bargap=0.35, hoverlabel=dict(bgcolor="white"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(gridcolor=GRILLE, zeroline=False)
    fig.update_yaxes(gridcolor=GRILLE, zeroline=False)
    return fig


# ---------------------------------------------------------------- en-tête
st.title("Découverte d'intentions par LLM-in-the-loop")
st.markdown(
    "2 033 commandes vocales en français (MASSIVE fr-FR) regroupées en intentions nommées **sans aucune étiquette**, "
    "par une boucle embeddings + UMAP + K-Means + LLM juge et nommeur (Qwen 2.5 14B). "
    "Les vraies intentions MASSIVE ne servent qu'à l'évaluation."
)
st.caption("Projet de fin d'année M2 MLSD, Université Paris Cité, 2025-2026 · I. Assebbane, R. Zekri, J. Kancel, "
           "J. Ben Dhaou · encadrant I. Keraghel · méthode adaptée de Hong et al., *Dial-In LLM* (arXiv 2412.09049)")

cmp_ = D.comparaison()
retenues = cmp_[cmp_["perimetre"] == "phrases retenues"]
tamis = retenues[retenues["methode"].str.startswith("Tamis LLM (")].iloc[0]
base79 = retenues[retenues["methode"].str.contains("repris du tamis") & retenues["methode"].str.startswith("UMAP")].iloc[0]
auto = retenues[retenues["methode"].str.contains("K choisi par")].set_index("methode")
m = D.mapping()
couverture = m["tag"].notna().mean()
juge = D.juge()
v1 = juge[juge["prompt"].str.startswith("v1")]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Phrases étiquetées", fr(couverture * 100, 1) + " %", help="Les autres restent dans le tamis")
c2.metric("Intentions créées", f"{m['tag'].nunique()}", help="Pour 59 intentions réelles dans MASSIVE")
c3.metric("Pureté des intentions", fr(tamis["purete"], 2),
          help="Part des phrases qui portent l'intention majoritaire de leur groupe. K-Means avec K choisi "
               "automatiquement : " + ", ".join(f"{fr(r['purete'], 2)} ({i.split('par ')[-1]})" for i, r in auto.iterrows())
               + ". À performance globale (ARI, NMI) comparable.")
if len(v1):
    acceptes = v1[v1["decision"] == "accepté"]
    rappel = (v1[v1["pur_attendu"]]["decision"] == "accepté").mean()
    c4.metric("Juge LLM : précision / rappel", f"{fr(acceptes['pur_attendu'].mean(), 2)} / {fr(rappel, 2)}",
              help=f"Précision : part des clusters acceptés qui sont vraiment purs. Rappel : part des clusters purs "
                   f"acceptés. Prompt du run, {len(v1)} clusters de test.")

onglets = st.tabs(["Résultats", "Taxonomie produite", "Le juge LLM", "Déroulé du run", "Méthode et limites"])

# ---------------------------------------------------------------- 1. résultats
with onglets[0]:
    st.subheader("Le tamis LLM fait-il mieux qu'un simple K-Means ?")
    g, d = st.columns([1, 1])
    metrique = g.radio("Métrique", ["ARI", "NMI", "purete"], horizontal=True,
                       format_func=lambda x: "Pureté" if x == "purete" else x)
    perimetre = d.radio("Phrases évaluées", ["phrases retenues", "corpus complet"], horizontal=True,
                        format_func=lambda x: "les 86 % étiquetées" if x == "phrases retenues"
                        else "100 % (reste rattaché au tag le plus proche)")
    t = cmp_[cmp_["perimetre"] == perimetre].sort_values(metrique)
    couleurs = [BLEU if s.startswith("Tamis LLM (") else "#9ec5f4" if s.startswith("Tamis") else
                ("#8f8d85" if i == "aucune" else "#d6d4cc") for s, i in zip(t["methode"], t["info_K"])]
    etiquettes = [m_ if "K =" in m_ else f"{m_}  (K={k})" for m_, k in zip(t["methode"], t["K"].astype(str))]
    fig = go.Figure(go.Bar(
        x=t[metrique], y=etiquettes, orientation="h", marker_color=couleurs,
        customdata=t["info_K"],
        text=[fr(v) for v in t[metrique]], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}<br>" + metrique + " : %{x:.3f}<br>information sur K : %{customdata}<extra></extra>"))
    fig.update_xaxes(range=[0, max(1.0, t[metrique].max() * 1.12)])
    st.plotly_chart(style(fig, 400), width="stretch")
    st.caption("Bleu : le tamis (bleu clair : le même sans la fusion des tags, pour mesurer son effet). Gris foncé : K-Means sans connaître K (K choisi par un critère interne), comme le tamis. "
               "Gris clair : K-Means à qui l'on donne K (le vrai nombre, ou celui du tamis), information indisponible en "
               "pratique. Baselines : moyenne sur 3 graines, mêmes embeddings Gemini.")

    a, b = st.columns(2)
    with a:
        st.markdown("**Lecture**")
        st.markdown(
            "- **K inconnu, comme en pratique : jeu égal avec K-Means.** Selon le critère qui choisit K, K-Means fait "
            "un peu mieux (Davies-Bouldin) ou un peu moins bien (silhouette) en ARI et NMI, dans la variance entre "
            "graines.\n"
            "- **Ce que le tamis apporte :** des intentions **nommées** (`ALARME_SUPPRIMER`...), plus **fines** (79 "
            "groupes, quand les critères internes en retiennent 20 à 40 pour 59 intentions réelles) et plus **pures**.\n"
            f"- À nombre de clusters égal, il gagne **+{fr(tamis['ARI'] - base79['ARI'], 2)} d'ARI**, grâce au "
            "**nommeur** qui fusionne les fragments d'une même intention. Sans cette fusion, l'ARI tombe sous toutes "
            "les baselines.\n"
            "- Non testé : un K-Means suivi d'un nommage par LLM. C'est la prochaine comparaison à faire.")
    with b:
        o = D.oracle()
        if o:
            st.markdown("**Plafond avec un juge parfait**")
            st.markdown(
                f"En remplaçant le LLM par un oracle qui connaît les vraies intentions : **{fr(o['couverture'] * 100, 1)} %** des "
                f"phrases étiquetées, ARI **{fr(o['ARI_phrases_retenues'])}**, pureté "
                f"**{fr(o['purete_phrases_retenues'])}**.\n\n"
                "Le LLM réel accepte donc plus de clusters qu'un juge parfait, au prix de la qualité ; et même un juge "
                "parfait laisse passer des erreurs, car il ne voit que 4 phrases par cluster.")
    with st.expander("Tableau complet"):
        st.dataframe(cmp_.rename(columns={"purete": "pureté", "ecart_type_ARI": "écart-type ARI"}),
                     hide_index=True, width="stretch")

# ---------------------------------------------------------------- 2. taxonomie
with onglets[1]:
    st.subheader("Les 79 intentions créées, et ce qu'elles contiennent vraiment")
    pt = D.purete_tags().sort_values("phrases", ascending=False)
    g, d = st.columns([2, 3])
    with g:
        tri = st.radio("Trier par", ["taille", "pureté croissante"], horizontal=True)
        vue = pt.sort_values("purete") if tri == "pureté croissante" else pt
        st.dataframe(
            vue.rename(columns={"tag": "intention créée", "purete": "pureté",
                                "intentions_massive": "intentions MASSIVE (top 3)"}),
            hide_index=True, width="stretch", height=460,
            column_config={"pureté": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1)})
    with d:
        choix = st.selectbox("Examiner une intention", vue["tag"].tolist(),
                             index=vue["tag"].tolist().index("LUMIERE_MODIFIER") if "LUMIERE_MODIFIER" in vue["tag"].values else 0)
        sous = m[m["tag"] == choix]
        comp = sous["intention_massive"].value_counts().sort_values()
        fig = go.Figure(go.Bar(x=comp.values, y=comp.index, orientation="h", marker_color=BLEU,
                               text=comp.values, textposition="outside", cliponaxis=False,
                               hovertemplate="%{y} : %{x} phrases<extra></extra>"))
        fig.update_layout(title=dict(text=f"{choix} : {len(sous)} phrases, réparties selon leur vraie intention",
                                     font=dict(size=14)))
        st.plotly_chart(style(fig, 80 + 34 * len(comp)), width="stretch")
        st.dataframe(sous[["phrase", "intention_massive", "iteration"]].rename(
            columns={"intention_massive": "intention MASSIVE", "iteration": "itération"}),
            hide_index=True, width="stretch", height=240)

    st.markdown("**Dans l'autre sens : comment chaque vraie intention a été découpée**")
    ec = D.eclatement()
    vraie = st.selectbox("Intention MASSIVE", ec["intention_massive"].tolist(),
                         index=ec["intention_massive"].tolist().index("calendar_set") if "calendar_set" in ec["intention_massive"].values else 0)
    rep = m[m["intention_massive"] == vraie]["tag"].fillna("(non étiquetée)").value_counts().sort_values()
    fig = go.Figure(go.Bar(x=rep.values, y=rep.index, orientation="h",
                           marker_color=[GRIS if i == "(non étiquetée)" else BLEU for i in rep.index],
                           text=rep.values, textposition="outside", cliponaxis=False,
                           hovertemplate="%{y} : %{x} phrases<extra></extra>"))
    st.plotly_chart(style(fig, 80 + 30 * len(rep)), width="stretch")
    st.caption("Exemple : calendar_set mélange rappels et événements dans MASSIVE ; le LLM les a séparés "
               "(CALENDRIER_CREE, REMINDER_CREER...). Les métriques le pénalisent, mais le découpage se défend.")

# ---------------------------------------------------------------- 3. juge
with onglets[2]:
    st.subheader("Le juge LLM sait-il reconnaître un cluster pur ?")
    if juge.empty:
        st.info("Aucun résultat d'évaluation du juge trouvé dans evaluation/results/.")
    else:
        st.markdown(
            "Clusters de 4 phrases construits à partir des vraies intentions : **purs**, **pollués faciles** (un intrus "
            "d'un autre domaine) et **pollués difficiles** (un intrus du même domaine, autre intention). "
            "Même prompt et même règle de décision que dans le pipeline ; Qwen 2.5 14B, 90 clusters par graine. "
            "Le graphique cumule les deux graines (60 clusters par type et par prompt).")
        agg = (juge.groupby(["prompt", "type"])["correct"].mean().reset_index())
        fig = go.Figure()
        for prompt, couleur in zip(sorted(agg["prompt"].unique()), [BLEU, ORANGE]):
            s = agg[agg["prompt"] == prompt].set_index("type").reindex(list(D.TYPES.values()))
            fig.add_bar(name=prompt, x=s.index, y=s["correct"], marker_color=couleur,
                        text=[f"{v:.0%}".replace("%", " %") for v in s["correct"]], textposition="outside",
                        cliponaxis=False, hovertemplate=prompt + "<br>%{x} : %{y:.0%} de bonnes décisions<extra></extra>")
        fig.update_layout(barmode="group", bargroupgap=0.12)
        fig.update_yaxes(range=[0, 1.12], tickformat=".0%", title="bonnes décisions")
        st.plotly_chart(style(fig, 360), width="stretch")

        lignes = []
        for (prompt, graine), s in juge.groupby(["prompt", "graine"]):
            acc = s[s["decision"] == "accepté"]
            purs = s[s["pur_attendu"]]
            lignes.append({"prompt": prompt, "graine": graine, "bonnes décisions": f"{s['correct'].sum()}/{len(s)}",
                           "précision (acceptés vraiment purs)": round(acc["pur_attendu"].mean(), 2) if len(acc) else None,
                           "rappel (purs acceptés)": round((purs["decision"] == "accepté").mean(), 2),
                           "réponses invalides": int((s["decision"] == "réponse invalide").sum()),
                           "latence moyenne (s)": round(s["latence_s"].mean(), 1)})
        st.dataframe(pd.DataFrame(lignes), hide_index=True, width="stretch")
        st.markdown(
            "- **v1** est trop sévère : il rejette la majorité des clusters purs, souvent pour une simple différence de "
            "variable (deux villes, deux devises).\n"
            "- **v2** (verdict par phrase, règle des variables explicitée) devait corriger cela. Résultat : **plus "
            "précis, mais encore plus sévère**, et pas meilleur au total. Un verdict par phrase donne trois occasions "
            "de dire non.\n"
            "- Une partie du désaccord vient de la finesse du découpage : le modèle distingue *pause* et *reprise* d'un "
            "livre audio, que MASSIVE regroupe. Sans taxonomie de référence fournie au modèle, une règle abstraite ne "
            "suffit pas à lui transmettre la bonne granularité.")

        st.markdown("**Parcourir les cas**")
        f1, f2, f3, f4 = st.columns([2, 1, 2, 1])
        p_ = f1.selectbox("Prompt", sorted(juge["prompt"].unique()))
        g_ = f2.selectbox("Graine", sorted(juge["graine"].unique()))
        t_ = f3.multiselect("Type", list(D.TYPES.values()), default=list(D.TYPES.values()))
        err = f4.toggle("Erreurs seulement", value=True)
        sel = juge[(juge["prompt"] == p_) & (juge["graine"] == g_) & (juge["type"].isin(t_))]
        if err:
            sel = sel[~sel["correct"]]
        st.caption(f"{len(sel)} cas")
        for _, c in sel.iterrows():
            icone = "✅" if c["correct"] else "❌"
            with st.expander(f"{icone} Cas {c['cas']} · {c['type']} · décision : {c['decision']} · « {c['central'][:70]} »"):
                phrases = [c["central"]] + list(c["extremes"])
                roles = ["centroïde"] + [f"extrême {i}" for i in range(1, len(phrases))]
                st.dataframe(pd.DataFrame({"rôle": roles, "phrase": phrases, "vraie intention": c["intentions"]}),
                             hide_index=True, width="stretch")
                if isinstance(c["verdicts"], list) and c["verdicts"]:
                    st.markdown("**Verdicts du modèle**")
                    for v in c["verdicts"]:
                        if isinstance(v, dict):
                            st.markdown(f"- {'✔' if v.get('meme_intention') else '✘'} *{v.get('phrase', '')}* : {v.get('raison', '')}")
                elif c["analyse"]:
                    st.markdown(f"**Raisonnement du modèle** (confiance {c['confiance']}) : {c['analyse']}")

# ---------------------------------------------------------------- 4. déroulé
with onglets[3]:
    st.subheader("31 itérations, environ 8 h de calcul")
    it = D.iterations()
    g, d = st.columns(2)
    with g:
        fig = go.Figure(go.Scatter(x=it["iteration"], y=it["phrases_restantes"], mode="lines+markers",
                                   line=dict(color=BLEU, width=2), marker=dict(size=8),
                                   hovertemplate="itération %{x} : %{y} phrases restantes<extra></extra>"))
        fig.update_layout(title=dict(text="Phrases encore non étiquetées au début de chaque itération", font=dict(size=14)))
        fig.update_yaxes(rangemode="tozero")
        st.plotly_chart(style(fig, 340), width="stretch")
    with d:
        fig = go.Figure(go.Bar(x=it["iteration"], y=it["phrases_extraites"], marker_color=BLEU,
                               hovertemplate="itération %{x} : %{y} phrases étiquetées<extra></extra>"))
        fig.update_layout(title=dict(text="Phrases étiquetées à chaque itération", font=dict(size=14)), bargap=0.25)
        st.plotly_chart(style(fig, 340), width="stretch")
    fig = go.Figure(go.Scatter(x=it["iteration"], y=it["k"], mode="lines+markers", line=dict(color=BLEU, width=2),
                               marker=dict(size=8), hovertemplate="itération %{x} : K = %{y}<extra></extra>"))
    fig.update_layout(title=dict(text="Nombre de clusters K choisi à chaque itération (compromis inertie / coût LLM)",
                                 font=dict(size=14)))
    fig.update_yaxes(rangemode="tozero")
    st.plotly_chart(style(fig, 300), width="stretch")
    st.caption("Arrêt à l'itération 31 : aucun cluster accepté. 277 phrases restent non étiquetées.")
    with st.expander("Paramètres par itération"):
        st.dataframe(it, hide_index=True, width="stretch")

# ---------------------------------------------------------------- 5. méthode
with onglets[4]:
    st.subheader("Méthode")
    st.markdown(
        "À chaque itération, sur les phrases non encore étiquetées :\n"
        "1. **UMAP** réduit les embeddings Gemini (3 072 dimensions), réglé par la trustworthiness ;\n"
        "2. **K-Means**, avec K = compromis entre l'inertie et le coût d'appel du LLM : "
        "`inertie(K)/inertie_max + λ·K/K_max`, λ mesuré à partir du temps de réponse ;\n"
        "3. le **juge** LLM reçoit la phrase centrale et les 3 plus éloignées de chaque cluster : pur ou impur ;\n"
        "4. le **nommeur** LLM nomme les clusters purs (`DOMAINE_ACTION`) en réutilisant les tags déjà créés ;\n"
        "5. les clusters impurs repartent au tour suivant.")
    st.markdown("**Ce qui vient du papier** : la boucle itérative elle-même et le nommage par LLM. "
                "**Ce que le projet change** : K choisi sans appel LLM, UMAP réoptimisé à chaque tour, juge sur 4 phrases "
                "au lieu de 20, fusion des tags pendant le run, adaptation au français, LLM local sans fine-tuning.")
    st.subheader("Limites")
    st.markdown(
        "- Un seul run complet du pipeline : pas de mesure de variance.\n"
        "- Le test de pureté sur 4 phrases est une heuristique.\n"
        "- La vérité terrain MASSIVE a son propre découpage, parfois plus grossier que celui du LLM, et un peu de bruit.\n"
        "- Le fine-tuning LoRA d'un modèle 7B a été exploré mais pas encore évalué.\n"
        "- MASSIVE est favorable aux embeddings (phrases courtes, domaines bien séparés) : le projet ne dit pas ce que "
        "vaudrait le tamis sur un corpus plus bruité, cas que le papier met en avant.")
    st.markdown("Code, données et scripts d'évaluation : [dépôt GitHub du projet](https://github.com/Johnathank78/llm-itl-intent-clustering).")
