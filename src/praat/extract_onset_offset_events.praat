selectObject: "TextGrid spoken3_modified"  ; 

tierIndex = 1
outputFile$ = "speech_events.txt"

numberOfIntervals = Get number of intervals: tierIndex

for i from 1 to numberOfIntervals
    label$ = Get label of interval: tierIndex, i
    if label$ = "sounding" or label$ = "sounding "
        startTime = Get start time of interval: tierIndex, i
        endTime = Get end time of interval: tierIndex, i
        appendFileLine: outputFile$, fixed$(startTime, 6) + "    " + "onset"
        appendFileLine: outputFile$, fixed$(endTime, 6) + "    " + "offset"
    endif
endfor

writeInfoLine: "Exported onset/offset events to ", outputFile$
