

import numpy as np
from scipy import stats

class RandomTree(object):

    def __init__(self, leaf_size = 1, verbose = False):
                self.leaf_size=int(leaf_size)
        pass

    def addEvidence(self, Xtrain, Ytrain):
        
        self.tree=self.buildtree(Xtrain, Ytrain)

    def buildtree(self, x, y):
        featurenum=int(x.shape[1]) 
        datanum=int(x.shape[0])
        if (datanum<=self.leaf_size):
            return [-1, stats.mode(y)[0], 0, 0]
        Ymax=np.max(y)
        Ymin=np.min(y)
        if (Ymax==Ymin):
            return [-1, Ymax, 0, 0]
        randfeatures=np.random.choice(featurenum,int(featurenum/2+1),replace=False)
        findthepair=False
        for fea in np.nditer(randfeatures):
            Fmax=np.max(x[:,int(fea)])
            Fmin=np.min(x[:,int(fea)])
            if (Fmax==Fmin):
                continue
            for i in range(int(1),int(3)):
                randnodeidx=np.random.choice(datanum, 2, replace=False)
                SplitVal=(x[int(randnodeidx[0]),int(fea)]+x[int(randnodeidx[1]),int(fea)])/2
                if (SplitVal==Fmax):
                    continue
                else:
                    findthepair=True
                    break
            if(findthepair==True):
                break
            else:
                findthepair=True
                SplitVal=(Fmax+Fmin)/2
                break
        if(findthepair==True):
            leftrange=(x[:,int(fea)]<=SplitVal)
            rightrange= ~leftrange
            left = self.buildtree(np.compress(leftrange,x,axis=0), np.compress(leftrange,y,axis=0))
            right = self.buildtree(np.compress(rightrange,x,axis=0), np.compress(rightrange,y,axis=0))
            root=[fea,SplitVal,1,int(len(left)/4)+1]
            return root+left+right
        else:
            leaf= self.makeMandatoryLeaf(y)
            return leaf

    def makeMandatoryLeaf(self,y):
        
        leaf=[-1,stats.mode(y)[0],0,0]
        return leaf

    def query(self,Xtest):
        
        tree=np.resize(self.tree,(int(len(self.tree)/4),4))
        fac=tree[:,0]
        spv=tree[:,1]
        lidx=tree[:,2].astype(np.int)
        ridx=tree[:,3].astype(np.int)
        datanum=int(Xtest.shape[0])
        feanum=int(Xtest.shape[1])
        
        nodes=np.zeros(datanum)
        nodes.dtype=np.int
        done=nodes<nodes
        t=done==done
        values=np.empty(datanum)
        auxexp = np.tile(np.arange(0, feanum), datanum)  

        while not np.all([done,t]):
            factors=np.take(fac,nodes)
            l=np.take(lidx,nodes)
            r=np.take(ridx,nodes)
            splitval=np.take(spv,nodes)
            nodeexp = np.repeat(factors, feanum)  
            res = np.resize(nodeexp == auxexp, (datanum, feanum))
            vbyfac=np.sum(res*Xtest,axis=1)
           
            values=values*done+~done*vbyfac
            goright=values>splitval
            nodes+=~goright*l+goright*r
            done=factors==-1
        return splitval

