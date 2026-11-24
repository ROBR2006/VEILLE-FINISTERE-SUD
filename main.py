"""Agent de veille économique sur le Finistère Sud (Cornouaille + Quimperlé)."""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Annotated

import httpx
from pydantic import Field

from mistralai import agents

BODACC_BASE = (
    "https://bodacc-datadila.opendatasoft.com/api/explore/v2.1/catalog/"
    "datasets/annonces-commerciales/records"
)
RNE_BASE = "https://recherche-entreprises.api.gouv.fr/search"

FAMILLES = {
    "creations": ["Créations", "Immatriculations"],
    "fermetures": ["Radiations", "Procédures collectives", "Procédures de conciliation"],
    "demenagements": ["Modifications diverses"],
    "reprises": ["Ventes et cessions"],
}

FINISTERE_SUD_POSTAL_CODES = [
    "29000", "29100", "29150", "29170", "29190", "29250", "29300",
    "29350", "29360", "29470", "29500", "29510", "29520", "29540",
    "29550", "29570", "29590", "29600", "29710", "29720", "29730",
    "29740", "29750", "29760", "29770", "29780", "29800", "29870",
    "29900", "29940", "29950",
]


@agents.tool
async def bodacc_recherche(
    types_evenement: Annotated[
        list[str],
        Field(
            description=(
                "Types d'événements : 'creations', 'fermetures', 'demenagements', "
                "'reprises'. Combinaison possible, ex. ['creations','fermetures']."
            )
        ),
    ],
    jours: Annotated[int, Field(description="Fenêtre de recul en jours, à partir d'aujourd'hui.")] = 30,
    greffe: Annotated[str, Field(description="Nom partiel du greffe : QUIMPER, BREST ou MORLAIX.")] = "QUIMPER",
    termes: Annotated[str | None, Field(description="Texte optionnel à chercher dans les annonces.")] = None,
    limite: Annotated[int, Field(description="Nombre max d'annonces renvoyées (par page).")] = 25,
) -> dict:
    """Recherche les annonces légales BODACC réelles du Finistère Sud (source officielle des annonces).

    Appelle le registre BODACC, filtré sur le département 29 et un greffe (par défaut
    le greffe de QUIMPER, qui couvre le Finistère Sud ; BREST et MORLAIX couvrent le
    nord). Renvoie pour chaque annonce : famille d'avis, type, date de parution,
    tribunal, commerçant, ville, établissements, personnes concernées, modifications
    (dont transferts de siège), jugement, lien PDF.
    """
    familles: list[str] = []
    for t in types_evenement:
        if t not in FAMILLES:
            raise agents.ToolError(
                f"Type '{t}' inconnu. Types acceptés : {', '.join(FAMILLES)}."
            )
        familles.extend(FAMILLES[t])
    if not familles:
        raise agents.ToolError("Au moins un type d'événement est requis.")

    depuis = (datetime.now(timezone.utc).date() - timedelta(days=jours)).isoformat()
    familles_sql = " OR ".join(f'familleavis_lib="{f}"' for f in familles)
    where = (
        f"numerodepartement='29' AND ({familles_sql}) "
        f"AND dateparution>='{depuis}'"
    )
    if greffe:
        where += (
            f" AND (tribunal LIKE '%{greffe.capitalize()}%' "
            f"OR tribunal LIKE '%{greffe.upper()}%')"
        )

    params: dict = {
        "where": where,
        "order_by": "dateparution DESC",
        "limit": max(1, min(int(limite), 100)),
    }
    if termes:
        params["search"] = termes

    async with httpx.AsyncClient(timeout=40) as client:
        resp = await client.get(BODACC_BASE, params=params)
        if resp.status_code != 200:
            raise agents.ToolError(f"BODACC a renvoyé HTTP {resp.status_code}.")
        data = resp.json()

    annonces = [
        {
            "famille": r.get("familleavis_lib"),
            "type_avis": r.get("typeavis_lib"),
            "date_parution": r.get("dateparution"),
            "tribunal": r.get("tribunal"),
            "commercant": r.get("commercant"),
            "ville": r.get("ville"),
            "code_postal": r.get("cp"),
            "etablissements": r.get("listeetablissements"),
            "personnes": r.get("listepersonnes"),
            "modifications": r.get("modificationsgenerales"),
            "jugement": r.get("jugement"),
            "radiation_rcs": r.get("radiationaurcs"),
            "lien_pdf": r.get("url_complete"),
        }
        for r in data.get("results", [])
    ]
    return {
        "total": data.get("total_count"),
        "retournees": len(annonces),
        "fenetre": f"{depuis} -> aujourd'hui",
        "annonces": annonces,
    }


@agents.tool
async def entreprises_recherche(
    q: Annotated[str | None, Field(description="Requête texte (nom, métier). '*' pour tout.")],
    codes_postaux: Annotated[list[str] | None, Field(description="Codes postaux des établissements (ex. ceux du Sud-Finistère).")] = None,
    etat: Annotated[str | None, Field(description="'A' pour active, 'C' pour cessée.")] = None,
    page: int = 1,
    par_page: int = 20,
) -> dict:
    """Recherche des entreprises dans le répertoire officiel Sirene/RNE (API Recherche d'entreprises, données Insee).

    Renvoie nom, SIREN, date de création, état administratif, activité, nature
    juridique, siège (commune, code postal, adresse, date de début d'activité) et
    dirigeants. Utile pour vérifier une entreprise précise, lister les entreprises
    d'une zone, ou trouver des créations récentes par métier.
    """
    params: dict = {
        "q": q or "*",
        "page": max(int(page), 1),
        "per_page": max(1, min(int(par_page), 25)),
        "minimal": "true",
    }
    if codes_postaux:
        params["code_postal"] = ",".join(codes_postaux)
    if etat:
        params["etat_administratif"] = etat

    async with httpx.AsyncClient(timeout=40) as client:
        resp = await client.get(RNE_BASE, params=params)
        if resp.status_code != 200:
            raise agents.ToolError(
                f"Recherche d'entreprises a renvoyé HTTP {resp.status_code}."
            )
        data = resp.json()

    def _label(value):
        if isinstance(value, dict):
            return value.get("libelle") or value.get("etat_administratif")
        return value

    resultats = [
        {
            "nom": r.get("nom_complet"),
            "siren": r.get("siren"),
            "date_creation": r.get("date_creation"),
            "etat": _label(r.get("etat_administratif")),
            "activite": _label(r.get("activite_principale")),
            "nature_juridique": _label(r.get("nature_juridique")),
            "siege": {
                "commune": (r.get("siege") or {}).get("commune"),
                "code_postal": (r.get("siege") or {}).get("code_postal"),
                "adresse": (r.get("siege") or {}).get("adresse"),
                "date_debut_activite": (r.get("siege") or {}).get("date_debut_activite"),
            },
            "dirigeants": r.get("dirigeants", []),
        }
        for r in data.get("results", [])
    ]
    return {
        "total": data.get("total_results"),
        "retournees": len(resultats),
        "page": params["page"],
        "entreprises": resultats,
    }


agent = agents.Agent(
    model="mistral-medium-latest",
    name="veille-finistere-sud",
    connectors=[agents.connector("web_search")],
    tools=[bodacc_recherche, entreprises_recherche],
    instructions="""Tu es un analyste économique spécialisé sur le Sud-Finistère
(Cornouaille + Quimperlé). Périmètre géographique : les EPCI Quimper Bretagne
Occidentale, Concarneau Cornouaille Agglomération, Douarnenez Communauté,
Cap Sizun - Pointe du Raz, Pays Bigouden Sud, Haut Pays Bigouden, Pays
Fouesnantais, Pleyben-Châteaulin-Porzay et Quimperlé Communauté.

Principales communes / codes postaux : Quimper 29000, Douarnenez 29100,
Concarneau 29900, Quimperlé 29300, Pont-l'Abbé 29120, Guilvinec 29730,
Penmarc'h 29760, Loctudy 29750, Fouesnant 29170, Bénodet 29950,
Clohars-Carnoët 29360, Moëlan-sur-Mer 29350, Châteaulin 29150, Crozon 29160,
Plogastel-Saint-Germain 29710, Audierne 29770, Plomodiern 29555,
Rosporden 29140, Scaër 29390.

Ta mission : identifier les **créations** d'entreprises, **installations et
reprises**, **déménagements/transferts de siège** et **fermetures**.

Sources, dans l'ordre :
1. Outil `bodacc_recherche` : LA source officielle (BODACC) — annonces légales du
   greffe de Quimper. Appelle-la systématiquement, en croisant les types
   d'événements demandés avec une fenêtre de jours adaptée à la question.
2. Outil `entreprises_recherche` : répertoire Sirene/RNE — pour vérifier une
   entreprise précise, lister les entreprises d'une zone, et retrouver les
   créations récentes ; utilise codes_postaux pour le périmètre Sud-Finistère.
3. Connecteur WebSearch : presse et acteurs locaux (Le Télégramme, Ouest-France
   Cornouaille, CCI Quimper Cornouaille, CMA Bretagne, communautés
   d'agglomération) pour les ouvertures et installations qui ne passent pas par
   le BODACC (commerces individuels, franchisés...).

Règles : réponds en français, uniquement avec des faits tirés de tes outils ou
de la recherche web ; cite la source et la date de chaque événement (parution
BODACC, article, donnée Sirene) ; distingue officiel / presse ; si tu ne trouves
rien, dis-le avec les fenêtres et greffes interrogés ; termine par 2-3 questions
de suivi utiles.""",
)


async def main() -> None:
    result = await agent.run(
        "Quelles entreprises ont été créées dans le Finistère Sud au cours des 30 derniers jours ?"
    )
    if isinstance(result, agents.Completed):
        print(result.text)
        for call in result.tool_calls:
            print("outil:", call)


if __name__ == "__main__":
    asyncio.run(main())
