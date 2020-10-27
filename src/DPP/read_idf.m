function idf = read_idf(fn,context)
  tmp = load(fn);
  idf = -ones(1,context.n);
  idf(tmp(:,1)) = tmp(:,2);
