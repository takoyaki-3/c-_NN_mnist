#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <random>
#include <cmath>

using namespace std;

typedef float matrixelement;

const int RAND = 1 << 20;
const float ZERO = 0.00001f;
const float ONE = 0.99999f;

typedef string cells_string;
typedef char cells_char;
typedef vector<cells_string> CellsTy;

CellsTy cells_split(cells_char* line, cells_char comma) {

	size_t len = strlen(line);
	line[len - 1] = comma;
	line[len] = '\0';
	int last = 0;

	CellsTy cells;
	string l;

	for (int i = 0; i < len; i++) {
		if (line[i] == '\r') continue;
		if (line[i] == '	') line[i] = comma;
		if (line[i] == comma) {
			cells.push_back(l);
			last = i;
			l.clear();
			continue;
		}
		l.push_back(line[i]);
	}

	return cells;
}

float sigmoidfunction(float x) {
	return 1 / (1 + expf(-x));
}

class matrix {
private:
public:
	// Matrix body
	matrixelement* mat;

	// Matrix size
	int n, m;

	// Init matrix
	matrix() {
		n = m = 1;
		mat = new matrixelement[n * m];
		memset(mat, 0, sizeof(matrixelement) * n * m);
	}
	matrix(int nn, int mm) {
		mat = new matrixelement[nn * mm];
		memset(mat, 0, sizeof(matrixelement) * nn * mm);
		n = nn;
		m = mm;
	}
	matrix(matrix* a) {
		mat = new matrixelement[a->n * a->m];
		n = a->n;
		m = a->m;
		memcpy(mat, a->mat, m * n * sizeof(matrixelement));
	}
	void newmat(int nn, int mm) {
		n = nn;
		m = mm;
		mat = new matrixelement[n * m];
		memset(mat, 0, sizeof(matrixelement) * n * m);
	}
	void copy(matrix* a) {
		n = a->n;
		m = a->m;
		delete[] mat;
		mat = new matrixelement[a->n * a->m];
		memcpy(mat, a->mat, m * n * sizeof(matrixelement));
	}
	void put(int i, int j, matrixelement e) {
		if (i >= n || j >= m || i < 0 || j < 0) return;
		mat[i * m + j] = e;
	}
	matrixelement get(int i, int j) {
		if (i >= n || j >= m || i < 0 || j < 0) return -1;
		return mat[i * m + j];
	}
	void random_m05to05() {
		std::random_device rnd;
		for (int i = 0; i < n; i++)
			for (int j = 0; j < m; j++)
				mat[i * m + j] = (matrixelement)(rnd() % RAND) / (matrixelement)RAND - 0.5f;
	}
	bool plus(matrix a) {
		if (a.n != n || a.m != m) return false;
		for (int i = 0; i < n; i++)
			for (int j = 0; j < m; j++)
				mat[i * m + j] += a.mat[i * m + j];
		return true;
	}
	bool minus(matrix a) {
		if (a.n != n || a.m != m) return false;
		for (int i = 0; i < n; i++)
			for (int j = 0; j < m; j++)
				mat[i * m + j] -= a.mat[i * m + j];
		return true;
	}
	bool dot(matrix a, matrix b) {
		if (a.m != b.n) return false;
		n = a.n;
		m = b.m;
		delete[] mat;
		mat = new matrixelement[n * m];
		memset(mat, 0, sizeof(matrixelement) * n * m);
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {
				for (int k = 0; k < a.m; k++) {
					put(i, j, get(i, j) + a.get(i, k) * b.get(k, j));
				}
			}
		}
		return true;
	}
	void output() {
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {
				printf("[%f]	", get(i, j));
			}
			cout << "\n";
		}
		cout << "\n";
	}
	void sigmoid() {
		for (int i = 0; i < n; i++)
			for (int j = 0; j < m; j++)
				mat[i * m + j] = sigmoidfunction(mat[i * m + j]);
	}
	void t_set(matrix* a) {
		n = a->m;
		m = a->n;
		delete[] mat;
		mat = new matrixelement[n * m];
		for (int i = 0; i < n; i++)
			for (int j = 0; j < m; j++)
				mat[i * m + j] = a->mat[j * a->m + i];
	}
	void scalar(matrixelement k) {
		for (int i = 0; i < n; i++)
			for (int j = 0; j < m; j++)
				mat[i * m + j] *= k;
	}
	void scalar_plus(matrixelement k) {
		for (int i = 0; i < n; i++)
			for (int j = 0; j < m; j++)
				mat[i * m + j] += k;
	}
	void scalar_division(matrixelement k) {
		for (int i = 0; i < n; i++)
			for (int j = 0; j < m; j++)
				mat[i * m + j] /= k;
	}
	bool multiplication(matrix* a) {
		if (n != a->n) return false;
		if (m != a->m) return false;
		for (int i = 0; i < n; i++)
			for (int j = 0; j < m; j++)
				mat[i * m + j] *= a->mat[i * m + j];
	}
	void clear() {
		delete[] mat;
	}
};

class NeuralNetwork {
public:
	int input_nodes;
	int hidden_nodes;
	int output_nodes;
	int hidden_layer;
	float learning_rate;

	vector<matrix> w;
	matrix wo;

	void save(string fname) {
		FILE* fp = fopen(fname.c_str(), "wb");
		fwrite(&input_nodes, sizeof(int), 1, fp);
		fwrite(&hidden_nodes, sizeof(int), 1, fp);
		fwrite(&output_nodes, sizeof(int), 1, fp);
		fwrite(&hidden_layer, sizeof(int), 1, fp);
		for (int i = 0; i < w.size(); i++) {
			fwrite(&w[i].n, sizeof(int), 1, fp);
			fwrite(&w[i].m, sizeof(int), 1, fp);
			fwrite(w[i].mat, sizeof(matrixelement), w[i].n * w[i].m, fp);
		}
		fwrite(&wo.n, sizeof(int), 1, fp);
		fwrite(&wo.m, sizeof(int), 1, fp);
		fwrite(wo.mat, sizeof(matrixelement), wo.n * wo.m, fp);
		fclose(fp);
	}
	void load(string fname) {
		FILE* fp = fopen(fname.c_str(), "rb");
		fread(&input_nodes, sizeof(int), 1, fp);
		fread(&hidden_nodes, sizeof(int), 1, fp);
		fread(&output_nodes, sizeof(int), 1, fp);
		fread(&hidden_layer, sizeof(int), 1, fp);
		for (int i = 0; i < hidden_layer; i++) {
			w.push_back(matrix());
			fread(&w[i].n, sizeof(int), 1, fp);
			fread(&w[i].m, sizeof(int), 1, fp);
			w[i].newmat(w[i].n, w[i].m);
			fread(w[i].mat, sizeof(matrixelement), w[i].n * w[i].m, fp);
		}
		fread(&wo.n, sizeof(int), 1, fp);
		fread(&wo.m, sizeof(int), 1, fp);
		wo.newmat(wo.n, wo.m);
		fread(wo.mat, sizeof(matrixelement), wo.n * wo.m, fp);
	}
	void init() {
		int bef = input_nodes;
		for (int k = 0; k < hidden_layer; k++) {
			w.push_back(matrix());
			w[k].newmat(hidden_nodes, bef);
			bef = hidden_nodes;
			w[k].random_m05to05();
		}
		wo.newmat(output_nodes, hidden_nodes);
		wo.random_m05to05();
	}
	void train(matrix* input, matrix* ans, float rate) {
		matrix final_outputs;
		vector<matrix> hidden_inputs;
		vector<matrix> hidden_outputs;
		hidden_inputs.push_back(matrix());
		hidden_inputs[0].copy(input);
		for (int i = 0; i < w.size(); i++) {
			hidden_outputs.push_back(matrix());
			hidden_outputs[i].dot(w[i], hidden_inputs[i]);
			hidden_outputs[i].sigmoid();
			hidden_inputs.push_back(matrix());
			hidden_inputs[i + 1].copy(&hidden_outputs[i]);
		}
		final_outputs.dot(wo, hidden_inputs[w.size()]);
		final_outputs.sigmoid();

		matrix output_errors(ans);
		output_errors.minus(final_outputs);

		output_errors.multiplication(&final_outputs);
		matrix s(ans->n, ans->m), t, u;
		s.scalar_plus(1.0f);
		s.minus(final_outputs);
		output_errors.multiplication(&s);
		t.t_set(&hidden_outputs[w.size() - 1]);
		u.dot(output_errors, t);

		final_outputs.clear();
		s.clear();

		wo.plus(u);
		t.clear();
		u.clear();


		for (int i = w.size() - 1; i >= 0; i--) {
			matrix hidden_errors;
			matrix wt;
			wt.t_set(&w[i]);
			hidden_errors.dot(wt, output_errors);
			output_errors.copy(&hidden_errors);

			matrix as(hidden_errors.n, hidden_errors.m);
			as.scalar_plus(1.0f);
			as.minus(hidden_outputs[i]);
			matrix at(&hidden_errors);
			at.multiplication(&hidden_outputs[i]);
			at.multiplication(&as);
			matrix au;
			au.t_set(&hidden_inputs[i]);
			matrix av;
			av.dot(au, av);
			av.scalar(learning_rate);

			w[i].plus(av);
			wt.clear();
			as.clear();
			at.clear();
			au.clear();
			av.clear();
		}

		for (int i = 0; i < w.size(); i++) {
			hidden_inputs[i].clear();
			hidden_outputs[i].clear();
		}
		hidden_inputs[w.size()].clear();
		output_errors.clear();
	}
	void query(matrix* input, matrix* output) {
		matrix x(input);
		for (int i = 0; i < w.size(); i++) {
			matrix o;
			o.dot(w[i], x);
			o.sigmoid();
			x.copy(&o);
			o.clear();
		}
		output->dot(wo, x);
		output->sigmoid();
		x.clear();
	}
};

void loadfile(string fname, vector<matrix>* inputs, vector<matrix>* outputs) {
	// Load file
	FILE* fp = NULL;
	fp = fopen(fname.c_str(), "r");

	while (!feof(fp))
	{
		char buf[1024 * 5];
		fgets(buf, 1024 * 5, fp);
		CellsTy cells = cells_split(buf, ',');
		if (cells.size() < 785) continue;

		matrix input(784, 1), output(10, 1);
		for (int i = 0; i < 784; i++) {
			input.put(i, 0, atof(cells[i + 1].c_str()));
		}
		input.scalar_division(255.0f);
		input.scalar(0.999f);

		output.scalar_plus(ZERO);
		output.put(atoi(cells[0].c_str()), 0, ONE);
		inputs->push_back(input);
		outputs->push_back(output);
//		output.output();
	}
	fclose(fp);
}

//#define MODE_USE_TRAINED_DATA
//#define MODE_OUTPUT_ON

int main() {

	// Setting neural network
	NeuralNetwork nn;
	nn.input_nodes = 784;
	nn.hidden_nodes = 200;
	nn.hidden_layer = 1;
	nn.output_nodes = 10;
	nn.learning_rate = 0.1;
	int epochs = 10;

	vector<matrix> inputs;
	vector<matrix> outputs;

	nn.init();

#ifdef MODE_USE_TRAINED_DATA
	nn.load("w");
#else
	loadfile("./sample_data/mnist_train_small.csv", &inputs, &outputs);
	cout << "Start training" << endl;
	for (int i = 0; i < epochs; i++) {
		cout << "epoch:	" << i << endl;
		for (int j = 0; j < inputs.size(); j++) {
			nn.train(&inputs[j], &outputs[j], 0.1);
		}
	}
	cout << "Training Done !" << endl;
	nn.save("w");
	cout << "Saved." << endl;
#endif

	inputs.clear();
	outputs.clear();

	loadfile("./sample_data/mnist_test.csv", &inputs, &outputs);

	int cou = 0;
	int sum = 0;
	for (int j = 0; j < inputs.size(); j++) {
		int ans = 0;
		for (int i = 1; i < 10; i++) {
			if (outputs[j].get(i, 0) > outputs[j].get(ans, 0)) ans = i;
		}

		nn.query(&inputs[j], &outputs[j]);
		int mindex = 0;
		for (int i = 1; i < 10; i++) {
			if (outputs[j].get(i, 0) > outputs[j].get(mindex, 0)) mindex = i;
		}
#ifdef MODE_OUTPUT_ON
		outputs[j].output();
		cout << "Answer:" << ans << endl;
		if (ans == mindex) cout << "success" << endl;
		else cout << "failed" << endl;
#endif
		if (ans == mindex) cou++;
		sum++;
	}
	cout << "Correct answer：" << cou << endl;
	cout << "Fialed answer：" << sum - cou << endl;
	cout << "Success rate：" << (float)cou / (float)(sum) << endl;
}