<!-- markdownlint-disable MD033 MD026-->
# Example usage

## How to draw a financial network?

* Loads the package.

```python
import tyrant as tt
```

* load the financial data`bank_specific_data`. see detail in `/libs/data`.  

**`tyrant.network.Data`**:  
tyrant.network.Data(filein_bank_specific_data, h_i_shock=None, R_ij=None, checks=True, clipneg=True, year='', p='', net='', r_seed=123)  

* **Return**: class Data.

* **Paramaters**:  
`filein_bank_specific_data`: str  
&emsp;&emsp;The file path containing the bank-specific data about banks.  
`h_i_shock`: None, array-like  
&emsp;&emsp;optional, the initial shock. see tt.creating_initial_shock().  
`R_ij`: None, float, array-like  
&emsp;&emsp;optional, defines how much the value of the asset A_ij(0) worths after bank j defaulted. R_ij := rho*A_ij(0), rho in [0,1].  
`checks`: True  
&emsp;&emsp;If True, a battery of checks is being run during object initialization.  
`clipneg`: True  
&emsp;&emsp;If True, negative values are set to zero.  
`year`: None, str  
&emsp;&emsp;optional, A label that can be provided or not, about the nature of the data. In this case, the year.  
`p`: None, float  
&emsp;&emsp;optional, the size of network connectivity. Useless for models.  
`net`: None, str  
&emsp;&emsp;optional, A label that can be provided or not, about the nature of the data. In this case, the net sample id.  
`r_seed`: int  
&emsp;&emsp;the random seed of *R*. for the function of matrix_estimation of library of 'NetworkRiskMeasures'. Default = 123.  

```python
path_bank_specific_data = '/Users/hehaoran/Desktop/bankdata/bank_specific_data(2010, 6, 30).csv'
h_i_shock = tt.creating_initial_shock(14,[1,2],0.05)
bank_data = tt.Data(filein_bank_specific_data=path_bank_specific_data,
                    h_i_shock=h_i_shock,
                    checks=False,
                    year='2010-06-30',
                    p='0.05',
                    net='Interbank Network')
```

* draw a financial network.

**`tyrant.drawing.Finetwork`**:  
tyrant.drawing.Finetwork(data, G=None, is_remove=True)

* **Return**: class Data.

* **Paramaters**:  
`data`:  
&emsp;&emsp;*Data* object, including all required. see tyrant.network.Data.  
`is_remove`:  
&emsp;&emsp;Remove all edges equal to 0. Default = True.

**`tyrant.drawing.Finetwork.draw`**:  
tyrant.drawing.Finetwork.draw(method='', h_i_shock=None, alpha=None, t_max=100, is_savefig=False, font_size=5, node_color='b', **kwargs)

* **Return**: a figure.

* **Paramaters**:  
`method`:  
&emsp;&emsp;optional, the color of nodes map to the important level of bank. i.e. {'dr','nldr','dc',...}. Default = 'dr'.  
`h_i_shock`:  
&emsp;&emsp;the initial shock. see tt.creating_initial_shock().  
`alpha`:  
&emsp;&emsp;optional, the parameter of Non-Linear DebtRank. Default = 0.  
`t_max`:  
&emsp;&emsp;the max number of iteration. Default = 100.  
`is_savefig`:  
&emsp;&emsp;optional, if True, it will be saved to the current work environment. otherwise, *plt.show()*.  
`font_size`:  
&emsp;&emsp;the size of the labels of nodes. Default = 5.  
`node_color`:  
&emsp;&emsp;the color of nodes. if *method* is not empty, the colors reflect the importance level.  
`**kwargs`:  
&emsp;&emsp;customize your figure, see detail in *networkx.draw*.  
<!-- &emsp;&emsp;... -->

```python
fn = tt.Finetwork(bank_data)
fn.draw(method='dr',is_savefig=True)
# Or customize the h_i_shock, like:
h_i_shock = tt.creating_initial_shock(bank_data.N(),[1,2],0.05)
fn.draw(method='dr',h_i_shock=h_i_shock)
```

![markdown](https://raw.githubusercontent.com/hehaoran-ori/Tyrant/master/docs/InterbankNetwork20100630.png)
