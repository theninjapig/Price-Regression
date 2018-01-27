import pandas as pd

def get_year(ffile='pp-complete.csv', fout='pre_train.csv', year="2014"):
    iter_csv = pd.read_csv(ffile, sep=',', iterator=True, chunksize=20000, names = range(1,17))
    df = pd.concat([chunk[chunk[3].str.contains(year)] for chunk in iter_csv]) 
    df.to_csv(fout, index=False, encoding='utf-8')


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        get_year(ffile=sys.argv[1])
        get_year(sys.argv[1],'pre_test.csv',"2015")
    else:
        print "please specify input"
        
    
    
    
