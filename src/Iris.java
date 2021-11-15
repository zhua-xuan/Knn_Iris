import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;
import org.apache.hadoop.util.StringUtils;

import java.io.*;
import java.net.URI;
import java.util.*;

public class Iris {
    public static class Rose implements WritableComparable <Rose> {
        Integer t_id;//测试数据id
        Double dis;//距离

        public Rose() {}

        public Rose(Integer t_id, Double dis) {
            super();
            this.t_id = t_id;
            this.dis = dis;
        }

        @Override
        public void write(DataOutput out) throws IOException {
            out.writeInt(t_id);
            out.writeDouble(dis);
        }

        @Override
        public void readFields(DataInput in) throws IOException {
            t_id=in.readInt();
            dis=in.readDouble();
        }

        @Override
        public int compareTo(Rose o) {//实现compare完成特定的排序
            if(o.t_id.equals(t_id))//先比较测试数据的id
                return dis.compareTo(o.dis);//再比较距离
            return t_id.compareTo(o.t_id);
        }

        @Override
        public boolean equals(Object o) {
            Rose i=(Rose) o;
            return i.compareTo(this)==0;//判断是否相等，考虑出现距离相等的情况
        }

        @Override
        public int hashCode() {
            return t_id.hashCode();//返回测数据的id的hash，将同一个id的发送到同一个reduce
        }
    }

    public static class  KnnMap extends Mapper<LongWritable,Text,Rose,Text> {

        private ArrayList<String> testData = new ArrayList<String>();//存储测试数据
        static Integer t_id;//记录标识测试数据的行号，将行号作为测试数据id
        String distanceway;//得到距离计算方式

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            //初始化 读取全局文件和全局参数
            t_id=1;
            Configuration conf = context.getConfiguration();
            distanceway= conf.get("distanceway","osjl");

            //逐行读取测试数据集文件
            URI[] URIs = Job.getInstance(conf).getCacheFiles();
            Path testPath = new Path(URIs[0].getPath());
            String testFileName = testPath.getName().toString();
            BufferedReader fis = new BufferedReader(new FileReader(testFileName));
            try{
                String line;
                while ((line = fis.readLine()) != null) {
                    testData.add(line);
                }
            }catch (IOException ioe) {
                System.err.println("Caught exception while parsing the cached file '" + StringUtils.stringifyException(ioe));
            }finally { fis.close(); }
        }

        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {

            //以训练数据txt作为输入数据
            String train_data=value.toString();
            String[] rc1=train_data.split(",");//文件是txt格式的，使用' '分割开
            String type=rc1[rc1.length-1];

            //计算该训练数据和各个测试数据的距离
            for(String i : testData){
                String[] rc2=i.split(",");
                Double dis = getdistance(rc1, rc2, distanceway);
                //key: 自定数据类型 测试数据行号，与当前训练数据距离 value:当前训练数据的所属类型
                context.write(new Rose(t_id, dis), new Text(type));
                t_id++;
            }
            t_id=1;//重置行号
        }

        private Double getdistance(String[] rc1, String[] rc2, String distanceway) {
            double sum=0;
            for(int m=0; m < rc2.length; m++){
                switch(distanceway){
                    case "osjl":sum+=Math.pow(new Double(rc1[m])-new Double(rc2[m]), 2);break;//欧氏距离计算
                    case "mhd":sum+=Math.abs(new Double(rc1[m])-new Double(rc2[m]));//曼哈顿距离计算
                }
            }
            if(distanceway.equals("osjl"))
                sum=Math.sqrt(sum);
            return sum;
        }
    }

    public static class KnnCombiner extends Reducer<Rose, Text, Rose, Text> {
        static int k;//全局参数k
        int j;//当前id剩余数据量
        Integer currentt_id;//当前id

        public void setup(Context context) {
            //初始化变量
            Configuration  conf = context.getConfiguration();
            k =  conf.getInt("k", 10);
            currentt_id=null;
            j=k;
        }

        @Override
        protected void reduce(Rose key, Iterable<Text> value, Context context)
                throws IOException, InterruptedException {

            if (currentt_id == null)//考虑到第一次运行
                currentt_id = key.t_id;//测试数据行号

            if (currentt_id != key.t_id) {//测试数据id变更就重置j和当前id
                j = k;
                currentt_id = key.t_id;
            }

            if (j != 0){//限制每个测试数据id只发送与他距离最近的前k条训练数据所属类型
                for (Text i : value) {
                    //考虑到距离相同的情况
                    j--;
                    context.write(key, i);
                    if (j == 0)
                        break;
                }
            }
        }
    }

    public static class KnnReduce extends Reducer<Rose, Text, Text, NullWritable>{

        static int k;//全局参数限制量top k
        String distanceway;//距离计算方式
        int j;//当前id剩余量
        HashMap<String, Integer> hm;//存储类标签和对应的次数
        Integer currentt_id;

        double right_count;//存储与检验文件相同的数量
        double false_count;//存储于检验文件不同的数量
        private ArrayList<String> labelData=new ArrayList<String>();//读取test_verify文件用于准确率accuracy的计算
        private ArrayList<String> testData = new ArrayList<String>();//存储测试数据

        public void setup(Context context) throws IOException, InterruptedException {
            Configuration  conf = context.getConfiguration();//初始化变量
            k =  conf.getInt("k", 10);
            distanceway= conf.get("distanceway","osjl");//默认距离计算方式欧式距离
            hm=new HashMap<String, Integer>();
            currentt_id=null;
            j=k;

            right_count=0;
            false_count=0;

            //逐行读取检验数据集文件
            URI[] verifyURIs = Job.getInstance(conf).getCacheFiles();
            Path patternsPath = new Path(verifyURIs[1].getPath());
            String verifyFileName = patternsPath.getName().toString();
            BufferedReader fis = new BufferedReader(new FileReader(verifyFileName));
            try{
                String line;
                while ((line = fis.readLine()) != null) {
                    labelData.add(line);
                }
            }catch (IOException ioe) {
                System.err.println("Caught exception while parsing the cached file '" + StringUtils.stringifyException(ioe));
            }finally {  fis.close(); }

            //逐行读取测试数据集文件
            URI[] URIs = Job.getInstance(conf).getCacheFiles();
            Path testPath = new Path(URIs[0].getPath());
            String testFileName = testPath.getName().toString();
            BufferedReader buf = new BufferedReader(new FileReader(testFileName));
            try{
                String line;
                while ((line = buf.readLine()) != null) {
                    testData.add(line);
                }
            }catch (IOException ioe) {
                System.err.println("Caught exception while parsing the cached file '" + StringUtils.stringifyException(ioe));
            }finally { buf.close(); }

            context.write(new Text("Sepal.Length,Sepal.Width,Petal.Length,Petal.Width,Predicted Value,True Value"), NullWritable.get());
        }

        @Override
        protected void reduce(Rose key, Iterable<Text> value, Context context)
                throws IOException, InterruptedException {


            if(currentt_id==null)//考虑第一次运行
                currentt_id=key.t_id;

            if(currentt_id!=key.t_id){//id变更修改当前id剩余量，改变当前id
                j=k;
                currentt_id=key.t_id;
            }

            if(j!=0) {
                //控制前k条数据进行运算
                for (Text i : value) {
                    //考虑距离一样的情况
                    Integer count = 1;
                    if (hm.containsKey(i.toString()))
                        count += hm.get(i.toString());
                    hm.put(i.toString(), count);//更新类别对应的计数
                    j--;

                    if (j == 0) {//id对应的前k条数据录入完毕，计算投票的结果
                        String newkey = getvalue(hm).trim();
                        String check = null, data = null;
                        if(labelData!=null&&labelData.size()>0){
                            check = labelData.get((int) (false_count + right_count));
                        }

                        if(testData!=null&&testData.size()>0){
                            data = testData.get((int) (false_count + right_count));
                        }

                        context.write(new Text(data + "," + newkey + "," + check), NullWritable.get());//写出类别预测结果
                        hm.clear();
                        if (check.equals(newkey))
                            right_count += 1;//计算预测效果和真实效果的差别数
                        else
                            false_count += 1;
                    }
                }
            }
        }

        private String getvalue(HashMap<String, Integer> hm) {
            List<Map.Entry<String,Integer>> list = new ArrayList(hm.entrySet());
            Collections.sort(list, (o1, o2) -> (o2.getValue() - o1.getValue()));
            return list.get(0).getKey();
        }

        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {
            //写出预测的结果和一些参数设置情况
            context.write(new Text("Distance calculation method:"+distanceway), NullWritable.get());
            context.write(new Text("k:"+k), NullWritable.get());
            context.write(new Text("accuracy:"+right_count/(right_count+false_count)*100+"%"), NullWritable.get());
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        String[] remainingArgs = new GenericOptionsParser(conf, args).getRemainingArgs();

        if(remainingArgs.length != 4){
            System.err.println("Usage: Iris_knn <in> <out> <test> <verify>");
            System.exit(1);
        }

        FileSystem fs = FileSystem.get(conf);
        if (fs.exists(new Path(remainingArgs[1])))
            fs.delete(new Path(remainingArgs[1]), true);

        try {
            Job job = Job.getInstance(conf, "iris knn");
            job.setJarByClass(Iris.class);

            job.setMapperClass(KnnMap.class);
            job.setMapOutputKeyClass(Rose.class);
            job.setMapOutputValueClass(Text.class);

            job.setCombinerClass(KnnCombiner.class);

            job.setReducerClass(KnnReduce.class);
            job.setOutputKeyClass(Text.class);
            job.setOutputValueClass(NullWritable.class);

            job.setInputFormatClass(TextInputFormat.class);
            job.setOutputFormatClass(TextOutputFormat.class);

            FileInputFormat.addInputPath(job, new Path(remainingArgs[0]));
            FileOutputFormat.setOutputPath(job, new Path(remainingArgs[1]));

            //获取test、test_verify文件的路径，并放到Cache中
            job.addCacheFile(new Path(remainingArgs[2]).toUri());
            job.addCacheFile(new Path(remainingArgs[3]).toUri());

            //提交作业
            int success = job.waitForCompletion(true) ? 0 : 1;
            //退出
            System.exit(success);

        }catch (Exception e) {
            e.printStackTrace();
        }
    }
}