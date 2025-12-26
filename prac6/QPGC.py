def QPGenericConstraints(G, g, A, C, b, d):
    """
    
    Convex optimization problem with equality and inequality constraints:
     
               min f(x)=1/2 x^T G X + g^T x     s.t.   A^T x = b i C^T x >= d

    Call it with:

    x = QPGenericConstrains(G, g, A, C, b, d)
    
    Notation:
    
    n = number of variables
    p = number of equality constants
    m = number of inequality constants
     
    Dimensions:   G -> n x n
                  g -> n x 1
                  A -> n x p
                  C -> n x m
                  b -> p x 1  
                  d -> m x 1  
    """

    def Newton_step(lamb0,dlamb,s0,ds):
        alp=1;
        idx_lamb0=np.array(np.where(dlamb<0))
        if idx_lamb0.size>0:
           alp = min(alp,np.min(-lamb0[idx_lamb0]/dlamb[idx_lamb0]))
         
        idx_s0=np.array(np.where(ds<0))
        if idx_s0.size>0:
           alp = min(alp,np.min(-s0[idx_s0]/ds[idx_s0]))
    
        return alp

    import numpy as np

    n=np.shape(G)[0]
    p=np.shape(A)[1]
    m=np.shape(C)[1]
   
    itermax=100
    x0=1.e-5*np.random.rand(n); gam0=np.ones(p); lamb0=np.ones(m); s0=np.ones(m);  #c.ini

    #KKT matrix
    Lamb=np.diagflat(lamb0)
    S=np.diagflat(s0) 
    
    KKT=np.bmat([[G,-A,-C,np.zeros((n,m))],[-A.T,np.zeros((p,p)),np.zeros((p,m)),np.zeros((p,m))],[-C.T,np.zeros((m,p)),np.zeros((m,m)),np.identity(m)],[np.zeros((m,n)),np.zeros((m,p)),S,Lamb]])
    
    #Residus i mesura complementaria
    ev=np.ones(m);
    rL=np.dot(G,x0)+g-np.dot(A,gam0)-np.dot(C,lamb0)
    rA=b-np.dot(A.T,x0)
    rC=s0-np.dot(C.T,x0)+d
    rslamb=np.dot(S,np.dot(Lamb,ev))
    mu=np.dot(s0.T,lamb0)/m;
    
    eps=1.e-16
    nit=0
    
    while np.linalg.norm(rL)>eps and np.linalg.norm(rA) > eps and np.linalg.norm(rC) > eps and  np.linalg.norm(mu) > eps and nit <= itermax:
         #Predictor -> affine values (maybe npn-feasible)
         indep=np.concatenate((rL,rA,rC,rslamb),axis=0)
         incr=np.linalg.solve(KKT,-indep)
         dx=incr[0:n]; dgam=incr[n:n+p]; dlamb=incr[n+p:n+p+m]; ds=incr[n+p+m:]
    
         #print(np.linalg.norm(ds+np.dot(np.dot(S,dlamb)+rslamb,np.linalg.inv(Lamb))))
         #print(np.linalg.norm(np.dot(G,dx)-np.dot(A,dgam)-np.dot(C,dlamb)+rL))
         #print(np.linalg.norm(-np.dot(A.T,dx)+rA))
         #print(np.linalg.norm(-np.dot(C.T,dx) - np.dot(np.dot(np.linalg.inv(Lamb),S),dlamb) +rC- np.dot(np.linalg.inv(Lamb),rslamb) ))
         
         #step length
         alp=Newton_step(lamb0,dlamb,s0,ds)
         
         #mesura complementaria (affine duality gap) i centering parameter
         mu2=np.dot((s0+alp*ds),(lamb0+alp*dlamb))/m
         sigm=(mu2/mu)**3
         
         #corrector
         dS=np.diagflat(ds)
         dLamb=np.diagflat(dlamb)
         indep=np.concatenate((rL,rA,rC,rslamb+np.dot(dS,np.dot(dLamb,ev))-sigm*mu*ev),axis=0)
         incr=np.linalg.solve(KKT,-indep)
         dx=incr[0:n]; dgam=incr[n:n+p]; dlamb=incr[n+p:n+p+m]; ds=incr[n+p+m:]
         
         #step lenght
         alp=Newton_step(lamb0,dlamb,s0,ds)
         #print("s0:");
         #print(np.min(s0),np.max(s0))
    
         #update
         alp=alp*0.95 
         x0=x0+alp*dx
         gam0=gam0+alp*dgam
         lamb0=lamb0+alp*dlamb
         s0=s0+alp*ds
         
         #update residus i mesura complementaria
         Lamb=np.diagflat(lamb0)
         S=np.diagflat(s0) 
         KKT=np.bmat([[G,-A,-C,np.zeros((n,m))],[-A.T,np.zeros((p,p)),np.zeros((p,m)),np.zeros((p,m))],[-C.T,np.zeros((m,p)),np.zeros((m,m)),np.identity(m)],[np.zeros((m,n)),np.zeros((m,p)),S,Lamb]])
         rL=np.dot(G,x0)+g-np.dot(A,gam0)-np.dot(C,lamb0)
         rA=b-np.dot(A.T,x0)
         rC=s0-np.dot(C.T,x0)+d
         rslamb=np.dot(S,np.dot(Lamb,ev))
         mu=np.dot(s0.T,lamb0)/m;
        
         #check prints
         nit+=1
         #print("\n%d %24.16e %24.16e %24.16e %24.16e"%(nit,np.linalg.norm(rL),np.linalg.norm(rA),np.linalg.norm(rC),np.linalg.norm(mu)));
         #print(np.min(x0),np.max(x0))

    return x0

