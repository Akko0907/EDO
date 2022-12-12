import numpy as np
import matplotlib.pyplot as plt

class EDO():
    def __init__(self,F: list, T: float, dt: float=0.5, kind: str='base'):
        if not isinstance(dt,(float,int)):
            raise TypeError("h must be an integer or float")
        if not isinstance(T,(float,int)):
            raise TypeError("T must be an integer or float")
        if not all(hasattr(val, "__call__") for val in F):
            raise TypeError("F must be a list of callables (functions)")
        if not isinstance(kind,str) or kind not in ['base','mod','rk2','rk4','special']:
            raise TypeError(f"kind must be a string between: {['base','mod','rk2','rk4','special']}")

        self.__F = F
        self.__h = dt
        self.__kind = kind 
        self.__T = T
        self.__N = int(T//dt)
        self.__sys_size = len(F)


    def __call__(self,init: np.ndarray,kind: str=None) -> tuple:
        if kind!=None:
            if not isinstance(kind,str) or kind not in ['base','mod','rk2','rk4','special']:
                raise TypeError(f"new_kind must be a string between: {['base','mod','rk2','rk4','special']}")
            self.__kind = kind

        sys_size = self.__sys_size
        N = self.__N
        
        pace = init.copy()
        ys = np.zeros((N+1,sys_size))
        ys[0] = pace
        
        for i in range(1,N+1):
            ys[i] = self.step(pace)  
            pace = ys[i]
        
        t = np.linspace(0,self.__T,N+1)
        ys = ys.T
        return t,*ys 


    @property
    def h(self) -> float:
        return self.__h
    @property
    def T(self) -> float:
        return self.__T
    @property
    def kind(self) -> str:
        return self.__kind
    

    @h.setter
    def h(self,new_h: float) -> None:
        if not isinstance(new_h,(float,int)):
            raise TypeError("new_h must be an integer or float")
        self.__h = new_h
        self.__N = int(self.T//self.h)
    @T.setter
    def T(self,new_T: float) -> None:
        if not isinstance(new_T,(float,int)):
            raise TypeError("new_T must be an integer or float")
        self.__T = new_T
        self.__N = int(self.T//self.h)
    @kind.setter
    def kind(self,new_kind: str) -> None:
        if not isinstance(new_kind,str) or new_kind not in ['base','mod','rk2','rk4','special']:
            raise TypeError(f"new_kind must be a string between: {['base','mod','rk2','rk4','special']}")
        self.__kind = new_kind
                

    def step(self,pace: np.ndarray) -> np.ndarray:
        
        sys_size = self.__sys_size
        F = self.__F

        if self.__kind=='base':
            k1 = np.array([self.h*F[i](*pace) for i in range(sys_size)])
            yi = pace + k1
            return yi
        
        elif self.__kind=="mod":
            k1 = np.array([self.h*F[i](*pace) for i in range(sys_size)])
            pace1 = pace + k1
            k2 = np.array([self.h*F[i](*pace1) for i in range(sys_size)])
            yi = pace + (k1 + k2)/2
            return yi

        elif self.__kind=="rk2":
            k1 = np.array([self.h*F[i](*pace) for i in range(sys_size)])
            pace1 = pace + k1/2
            k2 = np.array([self.h*F[i](*pace1) for i in range(sys_size)])
            yi = pace + k2
            return yi

        elif self.__kind=="rk4":
            k1 = np.array([self.h*F[i](*pace) for i in range(sys_size)])
            pace1 = pace + k1/2
            k2 = np.array([self.h*F[i](*pace1) for i in range(sys_size)])
            pace2 = pace + k2/2
            k3 = np.array([self.h*F[i](*pace2) for i in range(sys_size)])
            pace3 = pace + k3
            k4 = np.array([self.h*F[i](*pace3) for i in range(sys_size)])  
            yi = pace + (1/6)*(k1 + 2*k2 + 2*k3 + k4) 
            return yi

        elif self.__kind=="special":
            yi = pace.copy()
            for j in range(sys_size):
                yi[j] = yi[j] + self.h*F[j](*yi)
            return yi


