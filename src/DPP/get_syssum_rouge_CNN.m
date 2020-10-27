function [rouge_score, avg_sents] = get_syssum_rouge_CNN(testdocs, testdir, py_file, pos_sorting)

% get all human summaries
dest_ref_dir = fullfile(testdir, 'ref_summaries');
if ~exist(dest_ref_dir, 'dir')
    mkdir(dest_ref_dir);
    files = dir(testdir); %  [testdir '/d*']
    files = regexp({files.name}, '\d+', 'match');
    ids_notEmpty = ~cellfun(@isempty, files);
    files = files(ids_notEmpty);
    
    for folder = files
        folder = folder{1};
        find_sum = fullfile(testdir, folder, '*.sum');
        summaries = dir(find_sum{1});
        for sum = {summaries.name}
            sum_file = fullfile(testdir, folder, sum);
            copyfile(sum_file{1}, dest_ref_dir);
        end
    end
end

% generate system summaries (get predicted summary sentences)
dest_sys_dir = fullfile(testdir, 'sys_summaries');
if ~exist(dest_sys_dir, 'dir')
    mkdir(dest_sys_dir);
end

total_sents = 0;
for d = 1:numel(testdocs)
    yp = testdocs(d).Ypred;
    fn = fullfile(testdir, testdocs(d).name, [testdocs(d).name, '.txt']);
    fid = fopen(fn, 'r');
    
    total_sents = total_sents + length(yp);
        
    tline = fgets(fid);
    txt = cell(0);
    while ischar(tline)
        txt{end+1} = tline;
        tline = fgets(fid);
    end
    fclose(fid);
    
    if pos_sorting ~= 1
        yp = sort(yp);
    else
        % position-based sorting
        yp = sort(yp);
        pos = testdocs(d).pos;
        [val, idx] = sort(pos(yp));
        yp = yp(idx);
    end
    
    sents = [];
    for i=1:numel(yp)
        sents = [sents, txt{yp(i)}];
    end
    
    if 0 and length(sents) > 665
        sents = sents(1:665);
    end
    
    fn = fullfile(dest_sys_dir, [testdocs(d).name, '.sum']);
    fid = fopen(fn, 'w');
    fprintf(fid, '%s', sents);
    fclose(fid);
end
avg_sents = total_sents / numel(testdocs);

% compute Rouge-1,2,L scores
score_file = fullfile(testdir, 'rouge_scores.txt');
cmd_run = sprintf('python %s --system_dir %s --ref_dir %s --score_dir %s', ...
                        py_file, dest_sys_dir, dest_ref_dir, score_file);
status = system(cmd_run);
if status
    disp('problem on running a rouge score script! no score file is generated.');
    return;
end

fID = fopen(score_file, 'r');
rouge_score = fscanf(fID, '%f');
fclose(fID);

assert(length(rouge_score) == 12);   % 1,2,l,su4

