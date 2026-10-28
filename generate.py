import os
import numpy as np
import argparse

from pd_ameliore import pd_ameliore, charger_metadonnees

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate a completed ratings table.'
    )
    parser.add_argument(
        "--name",
        type=str,
        default="ratings_train.npy",
        help="Name of the npy of the ratings table to complete"
    )
    args = parser.parse_args()

    # DO NOT CHANGE THIS LINE
    print('Ratings loading...')
    table = np.load(args.name)
    print('Ratings Loaded.')

    # Metadonnees des films (namesngenre2.npy, une ligne par colonne de notes) : facultatives.
    # Cherchees a cote du fichier de notes, dans le dossier courant et a cote du script ;
    # si elles manquent ou ne correspondent pas, on retombe sur le PD sans metadonnees.
    dossiers = [os.path.dirname(os.path.abspath(args.name)), os.getcwd(),
                os.path.dirname(os.path.abspath(__file__))]
    meta = charger_metadonnees(table.shape[1], dossiers)
    print('Metadonnees des films :', 'trouvees' if meta is not None else 'absentes (PD sans metadonnees)')

    # PD ameliore : etages PD enchaines (films, utilisateurs, films) sur les residus d'une baseline a biais,
    # puis (si metadonnees) une regression ridge par utilisateur sur les genres et l'epoque des films.
    try:
        completed = pd_ameliore(table, verbose=True, metadonnees=meta if meta is not None else False)
    except Exception as erreur:
        if meta is None:
            raise
        print('Echec avec metadonnees (', repr(erreur), ') : on relance sans.')
        completed = pd_ameliore(table, verbose=True, metadonnees=False)

    # On ne remplace que les notes inconnues, les notes connues restent inchangees.
    missing = ~np.isfinite(table)
    table = np.asarray(table, dtype=np.float64).copy()
    table[missing] = completed[missing]

    # DO NOT CHANGE THIS LINE
    np.save("output.npy", table)
