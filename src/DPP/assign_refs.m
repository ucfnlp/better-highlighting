function [docs] = assign_refs(docs,context)
% load reference and gold summaries -- should not be run on test data!
  
  for i = 1:length(docs)
    directory = docs(i).dir;
    name = docs(i).name;
    base = [directory '/' name '/' name];
    if exist([base '.YY'], 'file')
        docs(i).Y = load([base '.YY']);
    elseif exist([base '.Y'], 'file')
        docs(i).Y = load([base '.Y']);
    end
        
    docs(i).refs = read_refs(directory,name,context);
  end
  
