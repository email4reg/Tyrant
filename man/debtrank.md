<!-- markdownlint-disable MD033 -->
# Example usage

## How to draw a financial network

* Loads the package

```python
import tyrant as tt
from tyrant import debtrank as dr
```

* load the financial data

```python
path_bank_specific_data = '/Users/hehaoran/Desktop/bankdata/bank_specific_data(2010, 6, 30).csv'
h_i_shock = dr.creating_initial_shock(14,[1,2],0.05)
bank_data = dr.Data(filein_bank_specific_data=path_bank_specific_data,
                    h_i_shock=h_i_shock,
                    checks=False,
                    year='2010-06-30',
                    p='0.05',
                    net='Interbank Network')
```

* the 'draw' method from class Finetwork can easy to draw a financial network

***tyrant.debtrank.Finetwork***:  
tyrant.debtrank.Finetwork(data, G=None, is_remove=True)

&emsp;***Return***: class.

&emsp;***Paramaters***:  
&emsp;&emsp;*data*: *Data* object, including all required. see tyrant.debtrank.Data.  
&emsp;&emsp;*is_remove*: Remove all edges equal to 0. Default is True.

***tyrant.debtrank.Finetwork.draw***:  
tyrant.debtrank.Finetwork.draw(font_size=5, width=0.8, node_color='#6495ED', method='dr', h_i_shock=None, t_max=100, is_savefig=False, **kwargs)

&emsp;***Return***: a figure.

&emsp;***Paramaters***:  
&emsp;&emsp;*font_size*: the size of the labels of nodes. Default is 5.  
&emsp;&emsp;*method*: optional, the color of nodes map to the important level of bank. i.e. {'dr','nldr'}. Default is 'dr'.  
&emsp;&emsp;*is_savefig*: optional, if True, it will be saved to the current work environment. otherwise *plt.show()*.  
&emsp;&emsp;*width*: Line width. Default is 0.8.  
&emsp;&emsp;*node_color*: the color of nodes. if *method* is not empty, the colors reflect the importance level.  
&emsp;&emsp;*h_i_shock*: the initial shock. see dr.creating_initial_shock().  
&emsp;&emsp;*t_max*: the max number of iteration. Default is 100.  
&emsp;&emsp;***kwargs*: customization, see detail in *networkx.draw*.  
&emsp;&emsp;...

```python
fn = dr.Finetwork(bank_data)
fn.draw(method='dr',is_savefig=True)
# Or customize the h_i_shock<np.ndarry>. like:
fn.draw(method='dr',h_i_shock=h_i_shock)
```

![markdown](https://raw.githubusercontent.com/hehaoran-ori/Tyrant/master/libs/InterbankNetwork20100630.png)
