NPFL103 Information Retrieval: Assignment 2
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

This is the specification of the second assignment required to complete the
Information Retrieval course (NPFL103) at the Faculty of Mathematics and
Physics, Charles University.

## 2. Goal and Objectives

## 2.1 Goal

To learn abbout available frameworks for Information Retrieval and use them to
deliver state-of-the-art results on the provided test collections (the same as
in the Assignment 1).

## 2.2 Objectives

 - Learn about publicly available information retrieval frameworks and choose
   one of them (without any restrictions)
 - Use the selected framework to setup/implement an IR system.
 - Optimize the system on the provided test collections (using the training
   topics)
 - Write a detailed report on your experiments.
 - Present your results during the course practicals.

## 3. Detailed specification

 A) Learn about publicly available information retrieval frameworks, for example:

     * Anserini: https://github.com/castorini/anserini
     * ColBERT: https://github.com/stanford-futuredata/ColBERT
     * Chroma: https://www.trychroma.com/
     * ElasticSearch: https://www.elastic.co/elasticsearch
     * FAISS: https://faiss.ai/
     * Qdrant: https://qdrant.tech/
     * Lucene: http://lucene.apache.org
     * Milvus: https://milvus.io/
     * OpenNIR: https://opennir.net/
     * OpenSearch: https://opensearch.org/
     * Pinecone: https://www.pinecone.io/
     * Pyserini: https://github.com/castorini/pyserini
     * Solr: https://solr.apache.org/
     * Tantivy: https://github.com/quickwit-oss/tantivy
     * Terrier: http://terrier.org/
     * weaviate: https://weaviate.io/
     * Whoosh: https://whoosh.readthedocs.io/en/latest/
     * Xapian/Omega: https://xapian.org/
     
       ... (any other)

    Explore, choose one, install it and setup a system to index the test
    collections.

 B) Design and evaluate a baseline system as similar as possible to the
    baseline system from Assignment 1:

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
    all the documents (in all SGML tags) and generate results for the
    training and test topics for both the languages:

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
    
 G) Prepare a short presentation (about 5 slides for up to 5 minutes)
    summarizing your approach, employed data structures, conducted experiments,
    results, decisions, unsolved issues etc., to be presented to the lecturer
    and your peers during the practicals.  
    
4. Package Contents

   With the only exception of this README file, the contents of this package is
   the same as for Assignment 1.

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

This assignment follows the Assignment 1 with the only difference that you are
required to use third-party (publicly available) IR framework instead of
developing your own.

## 9. Grading

You can earn up to 100 points: 

  0-65 points for the implementation of the system, experiments, results on the
       training topics, and the report,  
  0-30 points for the performance of the constrained system (run-1) on the test
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
