import pprint
from pinecone.grpc import GRPCClientConfig
from wine_guide import index_name, embedChunk, WineNote
from pc_client import create_pinecone_client

pc = create_pinecone_client()

def retrieval_augmented_prompt(query) -> list[WineNote]:
    query_results = index.query(vector = embedChunk(query), top_k=10, include_metadata=True, include_values=False)
    # pprint.pprint(f"query_results: {query_results}")
    contexts = [
        WineNote(
            note=match.metadata['chunk'], 
            source=match.metadata.get('source', 'unknown'),
            score=match.score
        )
        for match in query_results.matches
    ]

    return contexts

index = pc.Index(name=index_name, grpc_config=GRPCClientConfig(secure=False))

# query = "Dujac Clos de la Roche"
# Chorey Les Beaune 2021, Domaine Tollot Beaut, 60 euros  
# Chorey Les Beaune 2022, Domaine Arnoux Père et Fils, 58 euros  

# query = "Chorey Les Beaune 2021, Domaine Tollot Beaut"
# query = "Chorey Les Beaune"
# query = "Volnay"
# query = "Jean-Claude Boisset"
# query = "Les Damodes"
# query = """
# Chorey Les Beaune Vieilles vignes 2022, Rapet, 58 euros  
# Chorey Les Beaune 2021, Domaine Tollot Beaut, 60 euros  
# Chorey Les Beaune 2022, Domaine Arnoux Père et Fils, 58 euros  
# Auxey Duresses 2021, A. Paquet, 68 euros  
# Auxey Duresses 2020, L. Latour, 58 euros  
# Pommard 1er cru Les Bertins 2018, Domaine Chantal Lescure, 145 euros  
# Monthélie les Duresses 1er cru 2020, Domaine des Comtes Lafon, 108 euros  
# Monthélie 2020, Bouchard Père et Fils, 60 euros  
# Volnay 2019, J. Drouhin, 105 euros  
# Volnay 2021, Christophe Vaudoisey, 69 euros  
# Volnay 1er cru Clos des Chênes 2022, Christophe Vaudoisey, 148 euros  
# Volnay 1er cru Clos des Chênes 2019, Domaine Jean Michel Gaunoux & Fils, 106 euros  
# Volnay 1er cru Les Chevrets 2016, H. Boillot, 175 euros  
# Volnay 1er cru Les Fremiets 2012, Domaine Charles Pères et Fille, 126 euros  
# Chassagne Montrachet Vieilles vignes 2022, Marc Colin, 67 euros  
# Chassagne Montrachet 2019, Domaine Bernard Bachelet, 73 euros  """
query = """
Gevrey-Chambertin 1er Cru "Clos des Issarts" Monopole, Domaine Faiveley 2020, 290.00€

Gevrey-Chambertin 1er Cru "Combe aux Moines", Domaine Faiveley 2018, 230.00€

Gevrey-Chambertin 1er Cru "Lavaux Saint-Jacques", Domaine Raphaël Dubois, 149.00€

Gevrey-Chambertin 1er Cru "Lavaux Saint-Jacques", Domaine Jean-Luc Burguet, 220.00€

Gevrey-Chambertin 1er Cru "Champeaux", Domaine Olivier Guyot 2020, 175.00€

Gevrey-Chambertin 1er Cru "Aux Échézeaux", Domaine Taupenot-Merme 2020, 200.00€"""
# query = "Domaine Tollot Beaut"
# query = "Tollot Beaut"
# query = "Second derivative of a sinusoidal function"
# query = "Pernot Pere et Fils"
# query = "pernot pere et fils"
pprint.pprint(f"Query: {query}")
rag_result = retrieval_augmented_prompt(query)
pprint.pprint(rag_result)
