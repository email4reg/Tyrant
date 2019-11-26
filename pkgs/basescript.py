from collections import defaultdict
import sys


def _pretty_print_list_2_str(l):
    _str = ''
    if len(l) > 1:
        _str += '<'+str(l[0])+'>'
        for e in l[1:]:
            _str += ','+str(e)
        return _str
    return str(l[0])


class Option:
    def __init__(self,keywords,values=None,description=''):
        """
        In:
            keywords : a str, of a list of str. The keywords by which the option is identified.
            values : the list of values the option can take, if provided. 
                     By default, the first value in the list, is the default value.
                     If None is provided, then, by default it becomes ['False','True']
                     If 'ANY' is provided as part of the list, then, the option can take any arbitrary value.
        """
        if isinstance(keywords,str):
            self._keywords=[keywords]
        else:
            try:
                self._keywords=[str(kw) for kw in keywords]
            except:
                print("ERROR: keywords has not the appropriate format. A str, or a list of str's")
                assert False

        if values is None:
            self._values=['False','True']
        elif values == 'ANY':
            self._values=['ANY']
        else:
            try:
                self._values=[str(v) for v in values]
            except:
                print("ERROR: impossible to create list of values from values =",values)
                assert False

        self._description=str(description)

    def keywords(self):
        return self._keywords
    # def type(self):
        # return self._type
    def values(self):
        return self._values
    def default(self):
        try:
            return self._values[0]
        except:
            return None
    def name(self):
        return self._keywords[0]
    def has_keyword(self,keyword):
        if str(keyword) in self._keywords:
            return True
        return False
    def contains_value(self,value):
        if 'ANY' in self._values:
            return True
        if str(value) in self._values:
            return True
        return False
    def show(self,verbose=True):
        if verbose:
            print(_pretty_print_list_2_str(self._keywords)+':'+_pretty_print_list_2_str(self._values))
        else:
            print(self._keywords[0]+':'+_pretty_print_list_2_str(self._values))
    def __repr__(self):
        return 'Option:'+str(self._keywords[0])


_help_option=Option( ['help','h','-h','--help'],values=['False','True'], description='The help option. It shows this message')

class CommandLine:
    def __init__(self,options,extra_help_message=''):

        self._option_idx_2_mentioned=defaultdict(lambda: False)
        self._option_idx_2_chosen_value={}

        if isinstance(options,Option):
            self._options=[options]
        else:
            self._options=list(options)

        for option in self._options:
            assert isinstance(option,Option),'ERROR: option is not of type Option.'

        self._options.append( _help_option )

        self._extra_help_message=str(extra_help_message)

        self._keyword_2_option_idx={}
        for option_idx,option in enumerate(self._options):
            _value=option.default()
            if _value=='ANY':
                self._option_idx_2_chosen_value[option_idx]=None
            else:
                self._option_idx_2_chosen_value[option_idx]=_value
            for keyword in option.keywords():
                assert keyword not in self._keyword_2_option_idx.keys(),'ERROR: duplicate keywords in self._options'
                self._keyword_2_option_idx[keyword]=option_idx

        self._args=sys.argv[1:]
        for arg in self._args:

            try:
                keyword,value=arg.split('=')
            except:
                keyword=arg
                value=None

            if value=='ANY':
                print('ERROR: ANY is a reserved word, and it cannot be used as a value for any Option.')
                self.print_usage(1)

            option_idx=self.keyword_2_option_idx(keyword)
            option=self.keyword_2_option(keyword)

            self._option_idx_2_mentioned[option_idx]=True

            if value is None:
                if not option.contains_value('True'):
                    print('ERROR: value is None for the ',option,' which has a non empty, non boolean, value list.')
                    self.print_usage(1)
                self._option_idx_2_chosen_value[option_idx]='True'
            else:
                if not option.contains_value(value):
                    print('ERROR: value',value,'is not available in',option)
                    self.print_usage(1)
                self._option_idx_2_chosen_value[option_idx]=value
  
        if self._option_idx_2_mentioned[self.keyword_2_option_idx('help')]:
            self.print_usage(0)

    def keyword_2_option_idx(self,keyword):
        try:
            return self._keyword_2_option_idx[keyword]
        except:
            print('ERROR: keyword',keyword,'not macching any option in',self._options)
            self.print_usage(1)
    def keyword_2_option(self,keyword):
        return self._options[self.keyword_2_option_idx(keyword)]
    def __getitem__(self,keyword):
        option_idx=self.keyword_2_option_idx(keyword)
        value=self._option_idx_2_chosen_value[option_idx]
        if value is None:
            option=self._options[option_idx]
            print('ERROR: value is None for',option)
            self.print_usage(1)
        return value
    def print_usage(self,error_int=0,verbose=False):
        if error_int!=0:
            print('ERROR_STATUS',error_int)
        print('USAGE [./this.py]')
        for option in self._options:
            option.show(verbose=verbose)
        if self._extra_help_message!='':
            print('NOTICE:')
            print(self._extra_help_message)
        sys.exit(error_int)

##############################################################################
# Old version of Command Line Tools' #########################################
##############################################################################

class command_line_arguments:
    def __init__(self,options={'help':'Print this help.',None:['Default argument.','Default_Value','Alternative_Value_1','Alternative_Value_2']},extra_message=None):
        self.extra_message=str(extra_message)
        assert isinstance(options,dict)
        self._options=options
#        self._options_values={option:None for option in self._options.keys()}
        self._options_values={}
        for option in self._options.keys():
            self._options_values[option]=None
        #
        self._descriptions={}
        self._default_values={}
        self._alternative_values={}
        for option,_format in self._options.items():
            if isinstance(_format,list):
                self._descriptions[option]=_format[0]
                self._default_values[option]=_format[1]
                self._alternative_values[option]=_format[2:]
            else:
                self._descriptions[option]=_format
                self._default_values[option]=None
                self._alternative_values[option]=None
        for option,value in self._options_values.items():
            if value is None:
                self._options_values[option]=self._default_values[option]
            elif self._default_values[option] is not None or self._alternative_values[option] is not None:
                if 'ANY' not in self._alternative_values[option]:
                    assert value in self._default_values[option] or value in self._alternative_values[option]
        #
        for arg in sys.argv[1:]:
            if '=' in arg:
                try:
                    option,value=arg.split('=')
                except:
                    print('ERROR: Invalid argument:',arg)
                    self.print_usage(1)
                if option not in self._options.keys():
                    print('ERROR: Unknown option',option)
                    self.print_usage(1)
                else:
                    self._options_values[option]=value
            elif arg in ['help','h','?','-h','--help']:
                self.print_usage(0)
            else:
                self._options_values[None]=arg
        #
#        self._descriptions={}
#        self._default_values={}
#        self._alternative_values={}
#        for option,_format in self._options.items():
#            if isinstance(_format,list):
#                self._descriptions[option]=_format[0]
#                self._default_values[option]=_format[1]
#                self._alternative_values[option]=_format[2:]
#            else:
#                self._descriptions[option]=_format
#                self._default_values[option]=None
#                self._alternative_values[option]=None
#        for option,value in self._options_values.items():
#            if value is None:
#                self._options_values[option]=self._default_values[option]
#            elif self._default_values[option] is not None or self._alternative_values[option] is not None: 
#                if 'ANY' not in self._alternative_values[option]:
#                    assert value in self._default_values[option] or value in self._alternative_values[option]
    def print_usage(self,i):
        print('Usage:')
        print('./[this.py]')
        for option in self._options.keys():
            def_val=self._default_values[option]
            alt_vals=self._alternative_values[option]
            if alt_vals is not None:
                vals='=<'+def_val+'>,'+','.join(alt_vals)
            elif def_val is not None:
                vals='='+def_val
            else:
                vals=''
            print('['+str(option)+vals+']')
        print()
        print('Where:')
        for option in self._options.keys():
            print(str(option),':',self._descriptions[option])
        if self.extra_message is not None:
            print(self.extra_message)
        sys.exit(i)
    def options(self):
        return self._options.keys()
    def assigned_options(self):
        return self._options_values
    def __getitem__(self,x):
        return self._options_values[x]


if __name__ == "__main__":
    pass
