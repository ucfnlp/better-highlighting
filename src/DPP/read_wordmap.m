function map = read_wordmap(fn)
  if ~exist('fn','var')
    fn = '/zain/users/kulesza/dpp/data/mine2/shared/dict';
  end
  
  f = fopen(fn,'rt');
  c = textscan(f,'%[^\t]%*d');
  map = c{1};