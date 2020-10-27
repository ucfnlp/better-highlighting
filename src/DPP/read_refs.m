function refs = read_refs(directory, name, context)
  
  refs = [];
  files = dir([directory '/' name '/' name '*.sum.words']);

  for file = files'
    tmp = load([directory '/' name '/' file.name]);
    refs = [refs; sparse(tmp(:,1),tmp(:,3),1,1,context.n)];
  end
