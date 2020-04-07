import numpy as np

'''
SymbolSets: A list containing all the symbols (the vocabulary without blank)

y_probs: Numpy array with shape (# of symbols + 1, Seq_length, batch_size)
         Your batch size for part 1 will remain 1, but if you plan to use your
         implementation for part 2 you need to incorporate batch_size.

Return the forward probability of the greedy path (a float) and
the corresponding compressed symbol sequence i.e. without blanks
or repeated symbols (a string).
'''


def GreedySearch(SymbolSets, y_probs):
    # Follow the pseudocode from lecture to complete greedy search :-)
    forward_prob = 1
    forward_path_ = ""
    i = 0
    blankbefore = False
    for seq in range(y_probs.shape[1]):
        if seq == 0:
            best = np.argmax(y_probs[:, seq, :])
            forward_prob *= y_probs[best, seq, :]
            if best != 0:
                forward_path_ += SymbolSets[best-1]
        else:
            best = np.argmax(y_probs[:, seq, :])
            forward_prob *= y_probs[best, seq, :]
            if best == 0:
                blankbefore = True
            elif forward_path_[i] != SymbolSets[best-1]:
                    forward_path_ += SymbolSets[best-1]
                    i += 1
                    blankbefore = False
            elif blankbefore:
                forward_path_ += SymbolSets[best-1]
                i += 1
                blankbefore = False

    return forward_path_, forward_prob


##############################################################################


'''
SymbolSets: A list containing all the symbols (the vocabulary without blank)

y_probs: Numpy array with shape (# of symbols + 1, Seq_length, batch_size)
         Your batch size for part 1 will remain 1, but if you plan to use your
         implementation for part 2 you need to incorporate batch_size.

BeamWidth: Width of the beam.

The function should return the symbol sequence with the best path score
(forward probability) and a dictionary of all the final merged paths with
their scores.
'''


def BeamSearch(SymbolSets, y_probs, BeamWidth):
    # Follow the pseudocode from lecture to complete beam search :-)
    seq_len = y_probs.shape[1]
    # initiate the first time instance
    pathWithTerminalBlank, pathWithTerminalSymbol = [], []
    pathScore, blankPathScore = {}, {}
    # initiate path with terminal blank
    pathWithTerminalBlank.append("")
    blankPathScore[""] = y_probs[0, 0, :]
    # initiate path with terminal blank
    for symbol in range(len(SymbolSets)):
        pathScore[str(symbol)] = y_probs[symbol+1, 0, :]
        pathWithTerminalSymbol.append(str(symbol))

    # subsequent time step
    for seq in range(1, seq_len):
        # prune
        scorelist = [score for score in blankPathScore.values()]
        for score in pathScore.values():
            scorelist.append(score)
        scorelist.sort()
        scorelist.reverse()

        cutoff = scorelist[BeamWidth-1] if BeamWidth < len(scorelist) else scorelist[-1]
        PrunedPathsWithTerminalBlank, PrunedPathsWithTerminalSymbol = [], []
        PrunedBlankPathScore, PrunedPathScore = {}, {}
        # prune path with terminal blank
        for p in pathWithTerminalBlank:
            if blankPathScore[p] >= cutoff:
                PrunedPathsWithTerminalBlank.append(p)
                PrunedBlankPathScore[p] = blankPathScore[p]
        # prune with terminal symbol
        for p in pathWithTerminalSymbol:
            if pathScore[p] >= cutoff:
                PrunedPathsWithTerminalSymbol.append(p)
                PrunedPathScore[p] = pathScore[p]

        # extend paths with a blank
        UpdatedPathsWithTerminalBlank, UpdatedBlankPathScore = [], {}
        for p in PrunedPathsWithTerminalBlank:
            UpdatedPathsWithTerminalBlank.append(p)
            UpdatedBlankPathScore[p] = PrunedBlankPathScore[p] * y_probs[0, seq, :]
        for p in PrunedPathsWithTerminalSymbol:
            if p in UpdatedPathsWithTerminalBlank:
                UpdatedBlankPathScore[p] += PrunedPathScore[p] * y_probs[0, seq, :]
            else:
                UpdatedPathsWithTerminalBlank.append(p)
                UpdatedBlankPathScore[p] = PrunedPathScore[p] * y_probs[0, seq, :]

        # extend paths with a symbol
        UpdatedPathsWithTerminalSymbol, UpdatedPathScore = [], {}
        for p in PrunedPathsWithTerminalBlank:
            for c in range(len(SymbolSets)):
                new_p = p + str(c)
                UpdatedPathsWithTerminalSymbol.append(new_p)
                UpdatedPathScore[new_p] = PrunedBlankPathScore[p] * y_probs[c+1, seq, :]
        for p in PrunedPathsWithTerminalSymbol:
            for c in range(len(SymbolSets)):
                new_p = p if str(c) == p[-1] else p + str(c)
                if new_p in UpdatedPathsWithTerminalSymbol:
                    UpdatedPathScore[new_p] += PrunedPathScore[p] * y_probs[c+1, seq, :]
                else:
                    UpdatedPathsWithTerminalSymbol.append(new_p)
                    UpdatedPathScore[new_p] = PrunedPathScore[p] * y_probs[c+1, seq, :]

        # update probs and paths
        pathWithTerminalSymbol = UpdatedPathsWithTerminalSymbol
        pathWithTerminalBlank = UpdatedPathsWithTerminalBlank
        pathScore = UpdatedPathScore
        blankPathScore = UpdatedBlankPathScore

    # merge
    MergedPaths = pathWithTerminalSymbol
    mergedPathScores = pathScore

    for p in pathWithTerminalBlank:
        if p in MergedPaths:
            mergedPathScores[p] += blankPathScore[p]
        else:
            MergedPaths.append(p)
            mergedPathScores[p] = blankPathScore[p]

    # pick best path
    bestPath_ = ''
    bestPath = ''
    bestScore = 0
    FinalPathScore = {}

    for p in mergedPathScores.keys():
        if bestScore < mergedPathScores[p]:
            bestScore = mergedPathScores[p]
            bestPath_ = p
    for c in bestPath_:
        bestPath += SymbolSets[int(c)]

    for p in mergedPathScores.keys():
        new_p = ''
        for c in p:
            new_p += SymbolSets[int(c)]
        FinalPathScore[new_p] = mergedPathScores[p]

    return bestPath, FinalPathScore

