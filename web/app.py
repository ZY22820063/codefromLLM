import os

from flask import Flask, request, render_template, jsonify

from web.dto.Component import Component
from web.dto.SampleData import SampleData
from web.processor.smiles_processor import process_sample_data
from web.processor.tg_predict import tg_predict
from web.processor.tr_predict import tr_predict
from web.processor.ts_predict import ts_predict

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('form.html')


def validate_data(data):
    """
    验证和转换数据
    """
    try:
        temperature = float(data.get('temperature'))
        thickness = float(data.get('thickness'))
        time = float(data.get('time'))
        wavelength = float(data.get('wavelength'))
        diamine1_smiles = str(data.get('diamine1_smiles')).strip()
        diamine1_proportion = float(data.get('diamine1_proportion'))
        diamine2_smiles = str(data.get('diamine2_smiles')).strip()
        diamine2_proportion = float(data.get('diamine2_proportion'))
        dianhydride1_smiles = str(data.get('dianhydride1_smiles')).strip()
        dianhydride1_proportion = float(data.get('dianhydride1_proportion'))
        dianhydride2_smiles = str(data.get('dianhydride2_smiles')).strip()
        dianhydride2_proportion = float(data.get('dianhydride2_proportion'))

        return SampleData(temperature=temperature, thickness=thickness, time=time, wavelength=wavelength, diamines=[
            Component(diamine1_smiles, diamine1_proportion),
            Component(diamine2_smiles, diamine2_proportion),
        ], dianhydrides=[
            Component(dianhydride1_smiles, dianhydride1_proportion),
            Component(dianhydride2_smiles, dianhydride2_proportion),
        ]), 200
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid input data"}), 400


@app.route('/submit', methods=['POST'])
def submit():
    sample, status = validate_data(request.form)
    if status != 200:
        return jsonify(sample), status

    process_sample_data(sample)

    tg_predict_result = tg_predict(sample)
    tr_predict_result = tr_predict(sample)
    ts_predict_result = ts_predict(sample)

    return jsonify({"tg_predict_result": tg_predict_result, "tr_predict_result": tr_predict_result,
                    "ts_predict_result": ts_predict_result}), 200


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=True)
