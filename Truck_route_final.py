#!/usr/bin/env python
# coding: utf-8

# In[673]:


import json
import networkx as nx
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import numpy as np
import math
import matplotlib.pyplot as plt
import itertools
import copy


# In[674]:


get_ipython().system('pip install ACO-Pants')
import pants


# In[675]:


locations_df = pd.read_csv('SaO_Optilandia_resub_locations.csv')
links_df = pd.read_csv('SaO_Optilandia_resub_links.csv')


# In[676]:


locations_df.head()


# In[677]:


edgelist = links_df
nodelist = locations_df.copy(deep=True)


# In[678]:


#edgelist.head()
xx = nodelist.drop(columns=['is_depot' , 'is_customer','capacity','level'])


# In[679]:



xx


# In[ ]:





# In[680]:


locations_df.head(10)
Ids = locations_df.id
#print(Ids[1])


# In[681]:


links_df


# In[682]:


depot_lorries = json.load(open('SaO_Optilandia_resub_depot_lorries.json'))
depot_lorries


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[683]:


depot_locations = np.where(locations_df.is_depot)[0]
print(depot_locations)


# In[684]:


customer_location = np.where(locations_df.is_customer)[0]
print(customer_location)


# In[685]:


G = nx.Graph()
#calculate the euclidean distance for each link and add them to a list
id = 0
distance = []
for index in links_df.iloc[:,0]:
    x = (locations_df[locations_df['id']==links_df['id2'][id]].x)[links_df['id2'][id]] - (locations_df[locations_df['id']==index].x)[index]
    y = (locations_df[locations_df['id']==links_df['id2'][id]].y)[links_df['id2'][id]] - (locations_df[locations_df['id']==index].y)[index]
    z = math.sqrt((x**2) + (y**2))
    distance.append(z)
    id = id + 1
distance_array = np.array(distance)
type(distance_array)
distance_array.shape
#print(distance_array)

edges = []
G.add_nodes_from(range(len(locations_df)))
G.add_weighted_edges_from(edges)
pos = {k:v.values for k,v in locations_df[['x','y']].iterrows()}


# In[686]:


G.add_nodes_from(range(len(locations_df)))
G.add_weighted_edges_from(edges)
pos = {k:v.values for k,v in locations_df[['x','y']].iterrows()}


# In[ ]:





# In[ ]:





# In[687]:


plt.figure(figsize=(200,200))


# In[688]:


nodes = []

#creating nodes to extract x and y components.

for i in range(len(customer_location)):
    A = locations_df.iloc[customer_location[i],0]
    x = locations_df.iloc[customer_location[i],1]
    y = locations_df.iloc[customer_location[i],2]
    z = locations_df.iloc[customer_location[i],5]
    t = locations_df.iloc[customer_location[i],6]
    
    if(t/z < 0.2):
        nodes.append((x, y ,z ,t ,A , z/2-t))

ara = []
print(nodes)
#print(nodes[0][4])
def capacity(a ):
    for i in range ( len ( nodes)):
        if( nodes[i][4] == a):
            
            ara.append(nodes[i][5])
        






#euclidean costs not eculidian distance calculation b/w nodes.
def euclidean(a, b):
    return math.sqrt(pow(a[1] - b[1], 2) + pow(a[0] - b[0], 2))


# In[689]:


#storing id's of depot
print(depot_locations)
d1 = locations_df.iloc[depot_locations[0],1] , locations_df.iloc[depot_locations[0],2]
d2 = locations_df.iloc[depot_locations[1],1] , locations_df.iloc[depot_locations[1],2]
d3 = locations_df.iloc[depot_locations[2],1] , locations_df.iloc[depot_locations[2],2]
d4 = locations_df.iloc[depot_locations[3],1] , locations_df.iloc[depot_locations[3],2]
nd = []
beta= []
final = []
for i in range(len(nodes)):
    c = nodes[i][0] , nodes[i][1]
   
    beta.append ( min (euclidean(c ,d1 ),euclidean(c ,d2 ),euclidean(c ,d3 ),euclidean(c ,d4 )))
    nd.append((euclidean(c ,d1 ),euclidean(c ,d2 ),euclidean(c ,d3 ),euclidean(c ,d4 )  ))
    final.append(nd[i].index(beta[i]))
    



# In[690]:


arr=[]
arr1=[]
arr2=[]
arr3=[]
arr4=[]
for i in range ( len(nodes)):
    arr.append((final[i]+1 , nodes[i][4]))



# In[691]:


for i in range ( len(nodes)):
    xf,yf = arr[i]
    if(xf==1):
        arr1.append(yf)
    if(xf==2):
        arr2.append(yf)
    if(xf==3):
        arr3.append(yf)
    if(xf==4):
        arr4.append(yf)
        


# In[692]:


#print(arr1)
#print(arr2)
#print(arr3)
#print(arr4)


# In[693]:


cap = []
#cordinates extraction 
def cordinates(i):
    xc , yc = locations_df.iloc[i,1] ,locations_df.iloc[i,2]
    return (xc,yc)
#id extraction
def idx (x):
    for i in range (len(nodes)):
        if(nodes[i][0]==x):
            
            return(nodes[i][4] , -nodes[i][5])


# In[694]:


dep1=[]
#dep1.append(cordinates(117))
for i in range ( len(arr1)):
    dep1.append(cordinates(arr1[i]))
    

#Using AOC pants       
world = pants.World(dep1, euclidean)
solver = pants.Solver()
solution = solver.solve(world)
#print(solution.distance)
depA1 = solution.tour    # Nodes visited in order
#print(solution.path)    # Edges taken in order        
#print(cordinates(117))
#print(depA1)
idx(depA1[0][0])
cap.clear()
cap.append(117)
for i in range (len(depA1)):
    
    cap.append(idx(depA1[i][0]) )
cap.append(117)

    
    
print(cap)


# In[695]:


dep2=[]
cap2=[]
#dep2.append(cordinates(125))
for i in range ( len(arr2)):
    dep2.append(cordinates(arr2[i]))
    
   # print(cordinates(125),cordinates(arr2[i]))
#AOC pants


world = pants.World(dep2, euclidean)
solver = pants.Solver()
solution = solver.solve(world)
print(solution.distance)
depA2 = solution.tour    # Nodes visited in order
#print(solution.path)    # Edges taken in orde        
print(cordinates(125))
cap2.clear()
cap2.append(125)
for i in range (len(depA2)):
    cap2.append(idx(depA2[i][0]))
cap2.append(125)
print(cap2)


# In[696]:


print(dep2)


# In[697]:


dep3=[]
cap3=[]
cap3.append(372)
#dep3.append(cordinates(372))
for i in range ( len(arr3)):
    dep3.append(cordinates(arr3[i]))
    
    #print(cordinates(372),cordinates(arr2[i]))
world = pants.World(dep3, euclidean)
solver = pants.Solver()
solution = solver.solve(world)
print(solution.distance)
depA3 = solution.tour   # Nodes visited in order
#print(solution.path)    # Edges taken in order
for i in range (len(depA3)):
    cap3.append(idx(depA3[i][0]))
cap3.append(372)
print(cap3)


# In[698]:


dep4=[]
cap4=[]
cap4.append(522)
#dep4.append(cordinates(522))
for i in range ( len(arr4)):
    dep4.append(cordinates(arr4[i]))
    
    #print(cordinates(522),cordinates(arr2[i]))
#print(dep4)
world = pants.World(dep1, euclidean)
solver = pants.Solver()
solution = solver.solve(world)
print(solution.distance)
depA4 = solution.tour  # Nodes visited in order
#print(solution.path)    # Edges taken in order
for i in range (len(depA4)):
    cap4.append(idx(depA4[i][0]))
cap4.append(522)
print(cap4)


# In[699]:


print(cap ,cap2 ,cap3 ,cap4)


# In[700]:


links_df.head()
Id = links_df
#print(cap)

print(cap)
#print(cap2)


# In[701]:


data = {}
data2= {}
data['lorry_id' ] = ("117")
print(data)
data2['loc'] = cap
dataf = []
dataf.append(data)
dataf.append(data2)

print(dataf)


# In[702]:


data = {}
data2= {}
data['lorry_id'] = ("125")
print(data)
data2['loc'] = cap2
dataf2 = []
dataf2.append(data)
dataf2.append(data2)

print(dataf2)
#print(dataf)


# In[703]:


data = {}
data2= {}
data['lorry_id'] = ("372")
data2['loc'] = cap3
dataf3=[]
dataf3.append(data)
dataf3.append(data2)

print(dataf3)
#print(dataf)


# In[704]:


data = {}
data2= {}
data['lorry_id'] = ("522")
print(cap4)
data2['loc'] = cap4
dataf4=[]
dataf4.append(data)
dataf4.append(data2)

#print(dataf4)
#print(dataf)


# In[705]:


print(depot_locations)


# In[706]:


#datafinal.clear()

datafinal = dataf
datafinal.append(dataf2)
datafinal.append(dataf3)
datafinal.append(dataf4)
#datafinal.append(dataf2)

#print(dataf)

def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()


#print(json.dumps(dataf, default=np_encoder))
with open("Ans.json", "w") as outfile: 
    json.dump(json.dumps(dataf, default=np_encoder), outfile)
    
    


# In[707]:


f = open("Ans.json")
data = json.load(f)
data 


# In[ ]:





# In[708]:


#print(dataf)


# In[709]:


links_df


# In[710]:


depot_locations


# In[711]:


Ids


# In[712]:


edge_objects = []
for i in range ( len (links_df)):
    edge_objects.append((links_df.id1[i],links_df.id2[i],euclidean(cordinates(links_df.id1[i]), cordinates(links_df.id2[i]))))
#print(edge_objects)

toilets = [] # Mark two nodes (5 & 9) to be toilets
entrances = depot_locations # Mark two nodes (2 & 7) to be Entrances
common_nodes = [1,3,4,6,8] #all the other nodes

node_types = [(401, 'toilet'), (190, 'toilet'),
              (117, 'entrance'), (125, 'entrance'),(372 ,'entrance'),(522,'entrance')]

#create the networkx Graph with node types and specifying edge distances
G = nx.Graph()

for n,typ in node_types:
    G.add_node(n, type=typ) #add each node to the graph

for from_loc, to_loc, dist in edge_objects:
    G.add_edge(from_loc, to_loc, distance=dist) #add all the edges   


# In[713]:


def subset_typeofnode(G, typestr):
    '''return those nodes in graph G that match type = typestr.'''
    return [name for name, d in G.nodes(data=True) 
            if 'type' in d and (d['type'] ==typestr)]

#All computations happen in this function
def find_nearest(typeofnode, fromnode):

    #Calculate the length of paths from fromnode to all other nodes
    lengths=nx.single_source_dijkstra_path_length(G, fromnode, weight='distance')
    paths = nx.single_source_dijkstra_path(G, fromnode)

    #We are only interested in a particular type of node
    subnodes = subset_typeofnode(G, typeofnode)
    subdict = {k: v for k, v in lengths.items() if k in subnodes}

    #return the smallest of all lengths to get to typeofnode
    if subdict: #dict of shortest paths to all entrances/toilets
        nearest =  min(subdict, key=subdict.get) #shortest value among all the keys
        return(nearest, subdict[nearest], paths[nearest])
    else: #not found, no path from source to typeofnode
        return(None, None, None)


# In[714]:


#print(cap[0])
mojo = cap[1][0]
emp=[]
for i in range(len(cap)-2):
    #print(cap[i+1])
    mojo = cap[i+1][0]
    print(mojo)
    emp.append(mojo)
    print(find_nearest('entrance', fromnode=mojo))
    
#print(emp)
#find_nearest('entrance', fromnode=117)


# In[715]:



print(dataf)


# In[716]:


dataf


# In[717]:


#print(nodes)
print(nodes[0][4])


# In[718]:


hm1=[]
hm2=[]
hm3=[]
hm4=[]


for i in range (len(nodes)):
    #print(nodes[i][4])
    az=  find_nearest('entrance', fromnode=nodes[i][4])
    
    
    hm1.append(az[0])
    hm2.append(az[1])
    hm3.append(az[2])
    
    

#print(hm1)
#print(hm2)
#print(hm3)

#dnde['loc'] = {"lorry id = 117"}    
    
    
    
    


# In[719]:




hm1x=[]
hm2x=[]
hm3x=[]
hm4x=[]
for i in range (len(hm1)):
    if(hm1[i]==117):
        hm1x.append(hm3[i])
    if(hm1[i]==522):
        hm2x.append(hm3[i])
    if(hm1[i]==372):        
        hm3x.append(hm3[i])
    if(hm1[i]==125):        
        hm4x.append(hm3[i])    
        
  

#print(hm3)

hm1xss = hm1x[0]+hm1x[1]
hm1xss.remove(117)
hm1xss.remove(117)
#print(len(hm2x))
hm2xss = hm2x[0]+hm2x[1]+hm2x[3]+hm2x[4]+hm2x[5]
#print(hm1xss)
dataf.insert(2,hm1xss)   # run once out of order
dataf.insert(4,hm2xss)
print(len(hm3x))
hm3xss = hm3x[0]+hm3x[1]
hm4xss = hm4x[1]+hm4x[2]+hm4x[3]+hm4x[4]+hm4x[5]+hm4x[6]
dataf.insert(6,hm3xss)
dataf.insert(8,hm4xss)
print(dataf)


# In[720]:



print(dataf)


# In[ ]:





# In[721]:


print(dataf)
#type(dataf)


    
    
    


# In[ ]:





# In[ ]:





# In[722]:



def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()


#print(json.dumps(dataf, default=np_encoder))
with open("final.json", "w") as outfile: 
    json.dump(json.dumps(dataf, default=np_encoder), outfile)
    
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[723]:


nx.draw(G,with_labels=False,pos=pos,node_size=30)
#nx.draw(G,nodelist=depot_locations,node_color='g',node_size=50,alpha=1)
nx.draw_networkx_labels(G,pos,{i:i for i in depot_locations});
#nx.draw_networkx_nodes(G,pos,nodelist=customer_location,node_color='r',node_size=100,alpha=1);


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




