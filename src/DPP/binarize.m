function F = binarize(a, bins)
% if a is n x 1 and bins is 1 x b, returns a n x b+1 binary matrix
% indicating which bin each value falls in
  
  % binary version
  F = bsxfun(@le, a, [bins inf]) & bsxfun(@gt, a, [-inf bins]);
  
  % interpolated version
  % n = length(a);
  % b = length(bins);
  % F = zeros(n,b);
  % for i = 1:n
  %   pos = find(a(i) < [bins inf], 1);
  %   if pos == 1
  %     F(i,1) = 1;
  %   elseif pos == b+1
  %     F(i,end) = 1;
  %   else
  %     lg = a(i) - bins(pos-1);
  %     ug = bins(pos) - a(i);
  %     F(i,pos-1) = ug / (lg+ug);
  %     F(i,pos) = lg / (lg+ug);
  %   end
  % end
 