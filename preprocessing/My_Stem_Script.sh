###########STEPS FOR MYSTEM FILES PROCESSING###########

#######STEP_1########
##Cleaning CHI files for further preprocessing #######

# Loop over all .child files
for file in *.child
do
  # Remove speaker labels (*CHI, *CHX),
  # remove brackets []
  # remove punctuation
  sed 's/[*CHI:\t]//g;
       s/[*CHX:\t]//g;
       s/\[.*\]//g;
       s/[[:punct:]]//g' "$file" > "$file.cleaned"
done

#####STEP_2#########
##Run MyStem morphological analyzer in command prompt (Windows)

# for /F %i in ('dir /b *.child.cleaned') do mystem -i -n -d %i > %i.pos

#The command above loops over all cleaned child files
#Runs MyStem with the options:
# -i → add grammatical (POS) information
# -n → output one token per line
# -d → context-based disambiguation (removes homonymy)
# Output: POS-tagged files
# (sample.child.cleaned → sample.child.cleaned.pos)
# Output directory: C:\Users\Katya\Desktop\Dissertation\Data_Processing\my_stem
# Each .pos file should contain:
# 1 token per line; lemma + POS information; no CHAT markup or punctuation