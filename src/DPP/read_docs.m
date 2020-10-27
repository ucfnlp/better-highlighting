function docs = read_docs(directory, context, re_exp)
files = dir(directory);
files = regexp({files.name}, re_exp, 'match');
files = [files{:}];

docs = [];
for name = files
    docs = [docs read_doc(directory, name{1}, context)];
end
