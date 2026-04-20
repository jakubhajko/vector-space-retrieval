NPFL103 Information Retrieval: Assignment 1
===========================================

Lecturer: Pavel Pecina <pecina@ufal.mff.cuni.cz>
Web: https://ufal.mff.cuni.cz/courses/npfl103
Year: 2025/26

Contents of this document

 1. [Introduction](#1-introduction)
 2. [Goal and objectives](#2-goal-and-objectives)
 3. [Specification](#3-detailed-specification)
 4. [Package contents](#4-package-contents)
 5. [File formats](#5-file-formats)
 6. [Evaluation](#6-evaluation)
 7. [Submission](#7-submission)
 8. [Notes](#8-notes)
 9. [Grading](#9-grading)
10. [Plagiarism and joined work](#10-plagiarism-and-joined-work)
11. [Terms of use](#11-terms-of-use)

## 1. Introduction 

This is the specification of the first assignment required to complete the
Information Retrieval course (NPFL103) at the Faculty of Mathematics and
Physics, Charles University.

## 2. Goal and Objectives

### 2.1 Goal

The goal of this assignment is to get hands on implementation of vector space
model, inverted index, text preprocessing, system optimization, and
experimentation in Information Retrieval.

### 2.2 Objectives

 - Develop an experimental retrieval system based on vector space model and
   inverted index.
 - Experiment with methods for text processing, query construction, term and
   document weighting, similarity measurement, etc.
 - Optimize the system performance on the provided test collections (in English
   and Czech)
 - Write a detailed report on your system, experiments, and results 
 - Prepare a presentation and present your results during the course practicals.

## 3. Detailed Specification

 A) Use a programming language of your choice and implement an experimental
    retrieval system based on vector space model. For a given set of documents
    and a set of topics, the system is expected to rank the documents according
    to decreasing relevance to the topics. You are allowed to use standard
    libraries but not allowed to use any IR-specific libraries/packages/
    frameworks (e.g. search trees and hashes are OK, inverted index is not).
    You are also allowed to use third-party solutions for text preprocessing
    (such as tools for lemmatization, stemming, etc.). Your code must be
    fully executable via command line in a non-interactive mode. Implementations
    e.g. in Jupyter Notebook (Google Colab) will not be accepted.

    Expected usage:
    ./run -q topics.xml -d documents.lst -r run -o sample.res ...

    Where:
      -q topics.xml -- a file including topics in the TREC format (see Sec 5.2)
      -d documents.lst -- a file including document filenames (see Sec 5.1)
      -r run -- a string identifying the experiment (will be inserted in the
         result file as "run_id", see Sec 5.3)
      -o sample.res -- an output file (see Sec 5.3)
      ... (additional parameters specific to your implementation)

    You will have to consider solving the following issues:
     a) extraction of terms from the input data (data reading, tokenization,
        punctuation removal, ...)
     b) equivalence classing of terms (case folding, stemming, lemmatization,
        number normalization, ...)
     c) removing stopwords (frequency-based, POS-based, lexicon-based, ...)
     d) query construction (using title, description, narrative, ...)
     e) term weighting (boolean, natural, logarithm, logaverage, augmented,...)
     f) document frequency weighting (idf, probabilistic idf, ...)
     g) vector normalization (cosine, pivoted, ...) 
     h) similarity measure (cosine, BM25...)
     i) pseudo-relevance feedback
     j) query expansion (thesaurus-based, ...)
     
 B) Set up a baseline system (denoted as run-0) specified as follows:

    run-0 (baseline):
    - token delimiters: any sequence of whitespace and punctuation marks
    - term equivalence classes: no case normalization or other eq. classing
    - removing stopwords: no
    - query construction: all words from topic "title"
    - term frequency weighting: natural
    - document frequency weighting: none
    - vector normalization: cosine
    - similarity measure: cosine
    - pseudo-relevance feedback: none
    - query expansion: none
    
 C) Use the provided English and Czech test collections, index all tokens in
    all the documents (in all SGML tags) and generate results for the training
    and test topics for both the languages:

    Example usage:
    ./run -q topics-train_cs.xml -d documents_cs.lst -r run-0_cs -o run-0_train_cs.res

    Include at most 1000 top-ranked documents for each topic.

    Provide the following four files with your run-0 results:
    - run-0_train_cs.res for topics-train_cs.xml
    - run-0_test_cs.res for topics-test_cs.xml
    - run-0_train_en.res for topics-train_en.xml
    - run-0_test_en.res for topics-test_en.xml
    
    Evaluate the results for training topics (English and Czech) using the
    trec_eval tool.
 
    Example usage:
    ./eval/trec_eval -M1000 qrels-train_cs.txt run-0_train_cs.res
    ./eval/trec_eval -M1000 qrels-train_en.txt run-0_train_en.res
    
    The evaluation methodology is described in Section 6.
        
 D) Modify the baseline system by employing alternative/advanced methods for
    solving the issues and select the best combination (the system is to be
    denoted as run-1) which optimizes retrieval performance on the set of
    training topics (use Mean Average Precision as the main evaluation
    measure) for each of the two languages.
    
    You are allowed to use any third-party text processing/annotation tools
    (e.g. MorphoDiTa, available from http://ufal.mff.cuni.cz/morphodita, or
    UDPipe available from http://ufal.mff.cuni.cz/udpipe which are both useful
    for lemmatization and available for both English and Czech). You may use
    different tools/approaches for Czech and for English (e.g. stemming for
    Czech, lemmatization for English).

    Justify your decisions by conducting comparative experiments. For example,
    if you decide to use lemmas instead of word forms as terms, show that a
    system based on word forms performs worse on training topics. Or, if you
    decide to exclude some stop words from the index, show that such system
    performs better than a system indexing all words. Or, if you employ some
    query expansion technique, show that it improves results for the training
    topics.

    The only constraints are: i) the queries must be constructed from the topic
    *titles* only (i.e. you cannot use the topic description and topic
    narratives to construct the queries) ii) retrieval is completely automatic
    (no user intervention is allowed). You can also ignore (not index) text in
    certain SGML tags.
    
    run-1 (constrained):
    - token delimiters: ???
    - term equivalence classes: ???
    - case normalization: ???
    - removing stopwords: ???
    - query construction: based on topic titles
    - term weighting: ???
    - document frequency weighting: ???
    - vector normalization: ???
    - similarity measure: ???
    - pseudo-relevance feedback: ???
    - query expansion: ???

    Generate result files of this system for the training topics and test
    topics for both English and Czech. Include at most 1000 top-ranked
    documents for each topic.

    Provide the following four files with your run-1 results:
    - run-1_train_cs.res for topics-train_cs.xml
    - run-1_test_cs.res for topics-test_cs.xml
    - run-1_train_en.res for topics-train_en.xml
    - run-1_test_en.res for topics-test_en.xml
    
 E) Optionally, you are allowed to submit another system (denoted as run-2)
    without the restrictions (constraints in Sec 3, paragraph D) for extra
    points (see Sec 9 for details). You can use modified queries (e.g.
    formulated from topic descriptions and narratives) and use external data
    resources (e.g. a thesaurus).
 
    run-2 (unconstrained):
    - token delimiters: ???
    - term equivalence classes: ???
    - case normalization: ???
    - removing stopwords: ???
    - query construction: ???
    - term weighting: ???
    - document frequency weighting: ???
    - vector normalization: ???
    - similarity measure: ???
    - pseudo-relevance feedback: ???
    - query expansion: ???
    
    Again, generate result files for the training topics and test topics.
    Include at most 1000 top-ranked documents for each topic.
    
    If you decide to submit your run-2 results, provide the following four
    files with your run-2 results:
    - run-2_train_cs.res for topics-train_cs.xml
    - run-2_test_cs.res for topics-test_cs.xml
    - run-2_train_en.res for topics-train_en.xml
    - run-2_test_en.res for topics-test_en.xml
    
 F) Write a detailed report (in English or Czech/Slovak) describing details of
    your system (including building instructions, if needed), all the submitted
    runs (including instructions how the result files were generated) all
    conducted experiments and report their results on the training topics.
    Discuss the results and findings. Compare the results and approaches for
    English and Czech.
    
    For all experiments, report "map" (Mean Average Precision) as the main
    evaluation measure and "P_10" (Precision of the 10 first documents) as the
    secondary measure. For run-0 and run-1 with the training topics (both Czech
    and English) plot the Averaged 11-point averaged precision/recall graphs
    and include them in the report. 
    
 G) Prepare a short presentation (5-10 slides for ~5 minutes) summarizing
    your approach, employed data structures, conducted experiments, results,
    decisions, unsolved issues etc., to be presented to the lecturer and
    your peers during the practicals.
    
## 4. Package Contents

 - documents_cs -- a directory containing 221 files with the total of 81,735
     documents in Czech (see format description in Sec 5.1)

 - documents_en -- a directory containing 365 files with the total of 88,110
     documents in English (see format description in Sec 5.1)
   
 - documents_cs.lst -- a list of 221 filenames containing the Czech documents
 - documents_en.lst -- a list of 365 filenames containing the English documents

 - qrels-train_cs.xml -- relevance judgements for the Czech training topics
 - qrels-train_en.xml -- relevance judgements for the English training topics
     (see format description in Sec 5.4)
   
 - topics-train_cs.xml -- specification of the training topics in Czech
 - topics-train_en.xml -- specification of the training topics in English
 - topics-test_cs.xml -- specification of the test topics in Czech
 - topics-test_en.xml -- specification of the test topics in Czech

 - topics-test.lst  -- identifiers of the training topics
 - topics-train.lst -- identifiers of the test topics

 - sample-results.res -- example of result file (see Sec 5.3)

 - trec_eval-9.0.7.tar.gz -- the source code of the evaluation tool (see the
     included README for building instructions)

 - README.md -- this file
 
   NOTE: Relevance judgements for the (Czech and English) test topics are not
   provided to students)
 
## 5. File Formats

### 5.1 Document Format

The document format uses labeled bracketing, expressed in the style of SGML.
The SGML DTD used for verification at TREC/CLEF is included in the archive.
The files, however, are not guaranteed to be valid.

The philosophy in the formatting has been to preserve as much of the original
structure of the articles as possible, but to provide enough consistency to
allow simple decoding of the data.

Every document is bracketed by <DOC> </DOC> tags and has a unique document
number, bracketed by <DOCNO> </DOCNO> tags. The set of tags varies depending
on the language (see the DTD for more details). The Czech documents typically
contain the following tags (with corresponding end tags):
   
  <DOC>
  <DOCID>
  <DOCNO>
  <DATE>
  <GEOGRAPHY>
  <TITLE>
  <HEADING>
  <TEXT>

The English tag set is much richer than Czech. Generally all the textual
content (in any type of brackets) can be indexed, however this is also
something to (try to) optimize.

### 5.2 Topic Format

Topic specification uses the following markup:

  <top lang="xx">
  <num>
  ...
  </num>
  <title>
  ...
  </title>
  <desc>
  ...
  </desc>
  <narr>
  ...
  </narr>
  </top>

### 5.3 TREC Result Format 

The format of the system results (.res files) is 5 tab-separated columns:
  1. qid
  2. iter
  3. docno
  4. rank
  5. sim
  6. run_id

Example:
  10.2452/401-AH 0 LN-20020201065 0 0.9 baseline
  10.2452/401-AH 0 LN-20020102011 1 0.8 baseline

The important fields are "qid" (query ID, a string), "docno" (document number,
a string appearing in the DOCNO tags in the documents), "rank" (integer
starting from 0), "sim" (similarity score, a float) and "run_id" (identifying
the system/run name, must be same for one file). The "iter" (string) field
is ignored and unimportant for this assignment.

### 5.4 Query Relevance Format 

Relevance for each "docno" to "qid" is determined from train.qrels, which
contains 4 columns:
  1. qid
  2. iter
  3. docno
  4. rel

The "qid", "iter", and "docno" fields are as described above, and "rel" is
a Boolean (0 or 1) indicating whether the document is a relevant response for
the query (1) or not relevant response to the query (0).

Example:
 10.2452/401-AH 0 LN-20020201065 1
 10.2452/401-AH 0 LN-20020201066 0

 Document LN-20020201065 is relevant to topic 10.2452/401-AH.
 Document LN-20020201066 is not relevant to topic 10.2452/401-AH.
 
## 6. Evaluation

The official evaluation tool is provided in the "trec_eval-9.0.7.tar.gz"
package. Consult the "README" file inside the package for detailed
instructions.

Evaluation is performed by executing:
 ./eval/trec_eval -M1000 train-qrels.dat sample.res 

  where
    -M1000 specifies the maximum number of documents per query (do not change
       this parameter).
 
The tool outputs a summary of evaluation statistics:

  runid -- run_id
  num_q -- number of queries
  num_ret -- number of returned documents
  num_rel -- number of relevant documents
  num_rel_ret -- number of returned relevant documents
  map -- mean average precision (this is the main evaluation measure).
  ...
  iprec_at_recall_0.00 -- Interpolated Recall - Precision Averages at 0.00 recall
  ...
  P_10 -- Precision of the 10 first documents

For details see http://trec.nist.gov/pubs/trec15/appendices/CE.MEASURES06.pdf

There are also alternative implementations of the trec_eval tool available from
various sources. You can use them at your own risk. The official evaluation
will be done by the official TREC implementation included in the package.

## 7. Submission

Submission will be done by email. You are asked to attach a single (zip, tgz)
package containing:
  - the source code of your system and a "README" file with instructions how to
    build your system (if needed) and how to run the experiments.
  - The result files for training and test topics for at least run-0 and run-1
    both for English and for Czech.
  - The "report.pdf" file with your written report in the PDF format.
  - The "slides.pdf" file with a few slides in the PDF format, that you will
    use for presentation during practicals.
 
The run-0 result files must be obtained by running something like this:

  ./run -q topics-train_cs.xml -d documents_cs.lst -r run-0_cs -o run-0_train_cs.res
  ./run -q topics-train_en.xml -d documents_en.lst -r run-0_en -o run-0_train_en.res

which run the experiment and generates the training result files for English
and Czech.

  ./run -q topics-test_cs.xml -d documents_cs.lst -r run-0_cs -o run-0_test_cs.res
  ./run -q topics-test_en.xml -d documents_en.lst -r run-0_en -o run-0_test_en.res

For all the submitted systems (run-0, run-1, run-2) provide instructions how
to execute the experiments and obtain the results.

**Do not forget to generate the test result files and include them in the package
too!**

Keep the filenames of your report and presentation files as requested and put
them in the main (top) directory. 

Put all the result files in the main (top) directory too.

Do not include the original documents of the collection in the package (instead,
include a command to download the data from the given link) nor any
large intermediate results. Make the submission package small enough to be
delivered as an email attachment. 

## 8. Notes

In this assignment, you are asked to develop an experimental system. Efficiency
of your implementation (optimization for speed and memory use) will not be
evaluated. However, it is advised not to ignore this aspect of implementation
since efficient implementation will allow faster experimentation and designing
and running more experiments will (very likely) lead to better results. Proper
implementation of inverted index will make the experiments running much faster.

## 9. Grading

You can earn up to 100 points: 

  0-80 points for the implementation of the system, experiments, results on the
       training topics, and the report,  
  0-15 points for the performance of the constrained system (run-1) on the test
       topics,
   0-5 points for the oral presentation during course practicals.
   
In addition, you can get up to 10 extra points for the performance of the
unconstrained system (run-2) on the test topics (only if better than run-1).

## 10. Plagiarism and Joined Work

No plagiarism will be tolerated. The assignment is to be worked on on your own.
All cases of confirmed plagiarism will be reported to the Student Department.
You are only allowed to share your performance scores on the training data. 

## 11. AI Assistants

Using AI assistance is allowed but must be properly and fully akcnowledged in
the report (where and how it was used). Students are fully responsible for
their submissions.

## 12. Terms of Use

The data in this package (documents, topics, relevance assessment) may not be
distributed and/or shared in any way and can be used only for the purpose of
completing the assignments of the NPFL103 Information Retrieval at the Faculty
of Mathematics and Physics, Charles University.
