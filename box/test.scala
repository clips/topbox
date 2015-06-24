import scalanlp.io._;
import scalanlp.stage._;
import scalanlp.stage.text._;
import scalanlp.text.tokenize._;
import scalanlp.pipes.Pipes.global._;

import edu.stanford.nlp.tmt.stage._;
import edu.stanford.nlp.tmt.model.lda._;
import edu.stanford.nlp.tmt.model.llda._;

val modelPath = file("modelfolder");

println("Loading "+modelPath);
val model = LoadCVB0LabeledLDA(modelPath).asCVB0LDA;
val source = CSVFile("datafile.csv") ~> IDColumn(1);

val text = {
  source ~>                              // read from the source file
  Column(3) ~>                           // select column containing text
  TokenizeWith(model.tokenizer.get)      // tokenize with tokenizer above
}
 
val output = file(modelPath, source.meta[java.io.File].getName.replaceAll(".csv",""));
val dataset = LDADataset(text, model.termIndex);

println("Writing document distributions to "+output+"-document-topic-distributions-res.csv");
val perDocTopicDistributions = InferCVB0DocumentTopicDistributions(model, dataset);
CSVFile(output+"-document-topic-distributions-res.csv").write(perDocTopicDistributions);

// println("Writing topic usage to "+output+"-usage-res.csv");
// val usage = QueryTopicUsage(model, dataset, perDocTopicDistributions);
// CSVFile(output+"-usage-res.csv").write(usage);

// println("Estimating per-doc per-word topic distributions");
// val perDocWordTopicDistributions = EstimatePerWordTopicDistributions(
//   model, dataset, perDocTopicDistributions);
// CSVFile(output+"-document-word-topic-distributions.csv").write(perDocWordTopicDistributions);

// println("Writing top terms to "+output+"-top-terms.csv");
// val topTerms = QueryTopTerms(model, dataset, perDocWordTopicDistributions, numTopTerms=50);
// CSVFile(output+"-top-terms.csv").write(topTerms);

