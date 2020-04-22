import re

def warning(msg):
    #print("WARNING:", msg)
    pass

def convert_bio_to_spans(bio_sequence):
    spans = []  # (label, startindex, endindex)
    cur_start = None
    cur_label = None
    N = len(bio_sequence)
    for t in range(N+1):
        if ((cur_start is not None) and
                (t==N or re.search("^[BO]", bio_sequence[t]))):
            assert cur_label is not None
            spans.append((cur_label, cur_start, t))
            cur_start = None
            cur_label = None
        if t==N: continue
        assert bio_sequence[t] and bio_sequence[t][0] in ("B","I","O")
        if bio_sequence[t].startswith("B"):
            cur_start = t
            cur_label = re.sub("^B-?","", bio_sequence[t]).strip()
        if bio_sequence[t].startswith("I"):
            if cur_start is None:
                warning("BIO inconsistency: I without starting B. Rewriting to B.")
                newseq = bio_sequence[:]
                newseq[t] = "B" + newseq[t][1:]
                return convert_bio_to_spans(newseq)
            continuation_label = re.sub("^I-?","",bio_sequence[t])
            if continuation_label != cur_label:
                newseq = bio_sequence[:]
                newseq[t] = "B" + newseq[t][1:]
                warning("BIO inconsistency: %s but current label is '%s'. Rewriting to %s" % (bio_sequence[t], cur_label, newseq[t]))
                return convert_bio_to_spans(newseq)

    # should have exited for last span ending at end by now
    assert cur_start is None
    #spancheck(spans)
    return spans

def get_f1(preds,true):
    ## each is a list of lists with the string tags
    num_sent = 0
    num_tokens= 0
    num_goldspans = 0
    num_predspans = 0

    tp, fp, fn = 0,0,0
    for pred, true in zip(preds,true):
        spans = set()
        N = len(true)
        assert N==len(pred)
        num_sent += 1
        num_tokens += N

        goldspans = convert_bio_to_spans(true)
        predspans = convert_bio_to_spans(pred)

        num_goldspans += len(goldspans)
        num_predspans += len(predspans)

        goldspans_set = set(goldspans)
        predspans_set = set(predspans)

        tp += len(goldspans_set & predspans_set)
        fp += len(predspans_set - goldspans_set)
        fn += len(goldspans_set - predspans_set)
    prec = tp/(tp+fp) if (tp+fp)>0 else 0
    rec =  tp/(tp+fn) if (tp+fn)>0 else 0
    f1 = 2*prec*rec / (prec + rec) if (prec + rec) > 0 else 0
    print("F = {f1:.4f},  Prec = {prec:.4f} ({tp}/{tpfp}),  Rec = {rec:.4f} ({tp}/{tpfn})".format(
            tpfp=tp+fp, tpfn=tp+fn, **locals()))
    print("({num_sent} sentences, {num_tokens} tokens, {num_goldspans} gold spans, {num_predspans} predicted spans)".format(**locals()))
    return f1
