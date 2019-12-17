# Example usage

## draw financial network

### 1. load the bank data from <tt.Data>

```python
path_bank_specific_data = '/Users/hehaoran/Desktop/bankdata/bank_specific_data(2010, 6, 30).csv'
h_i_shock = dr.creating_initial_shock(14,[1,2],0.05)
bank_data = dr.Data(filein_bank_specific_data=path_bank_specific_data, h_i_shock=h_i_shock,
                    checks=False, year='2010-06-30', p='0.05', net='Interbank Network')
```

### 2. draw a financial network

the 'draw' method from class Finetwork can easy to draw a financial network
>***Paramaters***:  
&emsp;*method*: optional, the color of nodes map to the important level of bank. i.e {'dr': debtrank, 'nldr': non-linear debtrank}  
&emsp;*is_savefig*: optional, if True, If true, it will be saved to the current work environment. The default is show up now.  
...: # TODO detail

```python
fn = dr.Finetwork(bank_data)
fn.draw(method='dr',is_savefig=True)
# Or customize the h_i_shock<np.ndarry>. like:
fn.draw(method='dr',h_i_shock=h_i_shock)
```
