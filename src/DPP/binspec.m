function spec = binspec(name,n_bins)
  spec = {};
  for i = 1:n_bins+1
    spec = [spec {sprintf('%s %g',name,i)}];
  end