function doc = read_doc(directory, name, context)
  
  base = [directory '/' name '/' name];
  
  % load words
  tmp = load([base '.words']);
  N = max(tmp(:,1));
  max_length = max(tmp(:,2));
  txt = zeros(N,max_length);
  txt(sub2ind([N max_length],tmp(:,1),tmp(:,2))) = tmp(:,3);
  words = sparse(tmp(:,1),tmp(:,3),1,N,context.n);
    
  % pre-compute (normalized) tfidf
  tfidf = bsxfun(@times,words,context.idf);
  tfidf = bsxfun(@rdivide,tfidf,sqrt(sum(tfidf.^2,2)));
  tfidf(isnan(tfidf)) = 0; % in case of empty sentences
  tfidf(isinf(tfidf)) = 0;
  tfidf(tfidf<0) = 0;
  
  % load cost
  cost = load([base '.cost']);

  % load pos
  pos = load([base '.pos']);

  % build structure
  doc.dir = directory;
  doc.name = name;
  doc.N = N;
  doc.txt = txt; % N x max sentence length, contains word ids
  doc.words = words; % N x num words, contains counts
  doc.tfidf = tfidf;
  doc.cost = cost;
  doc.pos = pos;
