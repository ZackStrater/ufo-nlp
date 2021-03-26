K = range(1,50)
fig,ax = plt.subplots()
score=[]
inertia = []

for k in K:
    km = KMeans(k)
    km.fit(X_tfidf)
    score.append(-km.score(X_tfidf))
    inertia.append(km.inertia_)
    
ax.plot(K, inertia)

print(inertia)


'''
import itertools
maxk = 20
fig,ax = plt.subplots()
score=[]
wcss = np.zeros(maxk)

for k in K:
    km = KMeans(k)
    km.fit(X_tfidf)
    y = km.fit_predict(X_tfidf)
    score.append(-km.score(X_tfidf))

    for c in range(0, k):
        for i1, i2 in itertools.combinations([ i for i in range(len(y)) if y[i] == c ], 2):
            wcss[k] += sum(x[i1] - x[i2])**2
        wcss[k] /= 2
'''
