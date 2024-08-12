import numpy as np
from openbox import space as sp
from openbox import Optimizer
# Define Search Space
space = sp.Space()
cs = sp.ConditionedSpace()
x1 = sp.Real("x1", 0, 1, default_value=0)
x2 = sp.Real("x2", 0, 1, default_value=0)
x3 = sp.Real("x3", 0, 1, default_value=0)
space.add_variables([x1, x2, x3])

def Normalize(array):
    '''
    Normalize the array
    '''
    mx = np.nanmax(array)
    mn = np.nanmin(array)
    t = (array-mn)/(mx-mn)
    return t
    # return config['x1'] <= config['x2'] and config['x1'] * config['x2'] < 100
def ROC(config):
    a=np.loadtxt('')
    b=np.loadtxt('')
    c=np.loadtxt('')
    x1, x2, x3= config['x1'], config['x2'], config['x3']
    A=Normalize(a)
    B=Normalize(b)
    C=Normalize(c)
    d=x1*A+x2*B+x3*C
    list=d
    F = open('')
    for i in list:
        F.write(str(i)+'\n')
    F.close() 
    with open("","r",encoding="utf-8") as f:
        y_score=[]
        f=f.readlines()
        data=[i.split("\n")[0].split(" ") for i in f ]
        for line in data:
            y_score.append(float(line[0]))
    with open("","r",encoding="utf-8") as f:
        y_test=[]
        f=f.readlines()
        data=[i.split("\n")[0].split(" ") for i in f ]
        for line in data:
            y_test.append(int(line[0]))
    t=""
    return {'objectives': [t]}
opt = Optimizer(
    ROC,
    space,
    max_runs=50,
    surrogate_type='prf',
    task_id='quick_start',
)
history = opt.run()
print(history)

    
    
    
    
    
    