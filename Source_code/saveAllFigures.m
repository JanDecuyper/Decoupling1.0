function saveAllFigures(type)

h = get(0,'children');
for i=1:length(h)
  saveas(h(i), ['figure' num2str(i)], type);
end

end