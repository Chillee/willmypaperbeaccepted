
let closeModel = {
    lin1: tf.tensor([[0.1806, -0.7434],
    [0.6698, -0.5465],
    [0.7166, -0.2735],
    [0.4451, 0.3344],
    [-0.2999, -0.2087],
    [0.0744, -0.2855],
    [-0.7905, 0.2208],
    [-0.7536, 0.3097],
    [-0.4602, 0.1989],
    [0.6885, 0.1402],
    [0.6984, -0.6461],
    [0.1540, -0.1390],
    [-0.8042, -0.4016],
    [0.9019, 1.0477],
    [0.6438, 0.0442],
    [-0.5971, -0.1961],
    [0.3687, -0.2307],
    [-0.5273, -0.1064],
    [0.0028, 0.7800],
    [-0.5237, 0.3926],
    [-0.0049, -0.0500],
    [0.7471, 0.5462],
    [-0.2017, -0.6574],
    [-0.3410, -0.3331],
    [0.0109, 0.5915]]),
    lin1_bias: tf.tensor([0.2450, 0.1492, 0.7667, 0.3106, 0.6297, -0.1755, 0.7533, 0.9030,
        0.0815, 0.7612, -0.1955, -0.6267, -0.1502, 0.1181, 0.6338, -0.1813,
        0.4286, 0.0303, 0.7720, 0.2208, -0.3519, 0.4887, -0.2644, 0.3883,
        0.8705]),
    lin2: tf.tensor([[0.1418, -0.3549, -0.3071, -0.1214, 0.0782, -0.1546, 0.1194, 0.0058,
        0.2766, -0.4237, -0.3087, 0.0512, 0.0254, -0.3183, -0.3672, 0.3403,
        -0.1918, 0.4216, 0.3398, -0.0227, 0.1115, -0.4159, 0.4242, 0.2993,
        0.3252],
    [-0.0492, 0.1117, 0.4284, 0.1702, -0.1777, 0.1270, -0.2376, -0.3675,
    -0.5016, 0.4435, 0.1524, -0.0951, -0.2244, 0.1818, 0.3989, -0.2381,
        0.2673, -0.3415, -0.0921, -0.0750, -0.1038, 0.2071, -0.1675, -0.3102,
    -0.2800]]),
    lin2_bias: tf.tensor([0.2299, -0.3357]),
    mean: tf.tensor([5.9589491, 3.70114943]),
    variance: tf.tensor([1.08698477, 0.67368779])
};
let closerModel = {
lin1: tf.tensor([[ 0.1278,  0.4197],
        [-0.2713,  0.4527],
        [-0.3134,  0.8376],
        [-0.6156, -0.1563],
        [ 0.7305,  0.5634],
        [-0.8360, -0.3284],
        [ 0.8800,  0.3957],
        [ 0.5771, -0.1041],
        [ 0.1911, -0.6626],
        [-0.0324, -0.5128],
        [ 0.5033, -0.5072],
        [ 0.5479,  0.2231],
        [-0.2735, -0.4556],
        [ 0.3746,  0.1049],
        [-0.6090, -0.7193],
        [ 0.6060, -0.5328],
        [-0.6636,  0.4623],
        [-0.7950, -0.5062],
        [ 0.9432,  0.8604],
        [-0.4743, -0.1386],
        [-0.7678, -0.0053],
        [-0.7231, -0.2762],
        [-0.6727, -0.2312],
        [-0.7622, -0.1423],
        [-0.7389,  0.7662]]), lin1_bias: tf.tensor([-0.4117,  0.4032,  0.1746,  0.1324, -0.2956,  0.0213,  0.2622,  0.4225,
         0.1681, -0.5641,  0.1174,  0.1992,  0.1707,  0.4754, -0.3996, -0.1836,
         0.3347,  0.2237,  0.1138,  0.1943,  0.3414,  0.1559, -0.0896,  0.4859,
         0.7746]), lin2: tf.tensor([[ 0.1714,  0.3481,  0.2427,  0.0725, -0.2851,  0.1473, -0.3713, -0.1671,
         -0.3894,  0.1797, -0.1595, -0.3190,  0.2935, -0.3681,  0.0472, -0.1759,
          0.1386,  0.1563, -0.2014,  0.0152,  0.1115,  0.0212,  0.2095,  0.0149,
          0.3155],
        [-0.2898, -0.1856, -0.3542, -0.3884,  0.1448, -0.3663,  0.3769,  0.2610,
          0.3639, -0.2761,  0.4096,  0.4145, -0.3163,  0.4426, -0.0150,  0.5222,
         -0.0434, -0.3458,  0.2995, -0.2795, -0.3367, -0.1799, -0.2937, -0.3259,
         -0.1844]]), lin2_bias: tf.tensor([0.0192, 0.3398]),
    mean: tf.tensor([5.98984303, 3.72114497]),
    variance: tf.tensor([0.94264845, 0.63507459])
        };




function apply_linear(x, lin_weight, lin_bias) {
    let lin = lin_weight.concat(lin_bias.expandDims(axis = 1), axis = 1);
    return tf.dot(x.concat(tf.ones([x.shape[0], 1]), axis = 1), lin.transpose());
}

function get_prob(reviews, model) {
    let norm_reviews = reviews.sub(model.mean).div(model.variance);
    let x = apply_linear(norm_reviews, model.lin1, model.lin1_bias);
    x = tf.relu(x);
    x = x.mean(axis = 0).expandDims(axis = 0);
    x = apply_linear(x, model.lin2, model.lin2_bias);
    x = tf.softmax(x, axis = 1);

    return x.arraySync()[0][1];
}

let ratings = document.getElementsByClassName('rating');
let confs = document.getElementsByClassName('confidence');


function getData() {
    let num_ratings = [];
    for (let rating of ratings) {
        num_ratings.push(parseFloat(rating.noUiSlider.get()));
    }
    let num_confs = [];
    for (let conf of confs) {
        num_confs.push(parseFloat(conf.noUiSlider.get()));
    }
    const data = tf.tensor([num_ratings, num_confs]).transpose();
    return data;
}
function initSliders() {
    for (let rating of ratings) {
        if (rating.noUiSlider) continue;
        noUiSlider.create(rating, {
            start: [6],
            step: 1,
            range: {
                'min': 1,
                'max': 10
            },
            pips: {
                mode: 'positions',
                values: [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                density: 10,
                stepped: true
            }
        });

    };
    for (let conf of confs) {
        if (conf.noUiSlider) continue;
        noUiSlider.create(conf, {
            start: [4],
            step: 1,
            range: {
                'min': 1,
                'max': 5
            },
            pips: {
                mode: 'positions',
                values: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                density: 20,
                stepped: true
            }
        });
    }
    function updFunction() {
        const data = getData();
        const prob = (get_prob(data, closeModel) + get_prob(data, closerModel))/2.0;
        updatePage(prob);
    }
    for (let rating of ratings) {
        rating.noUiSlider.on('update', updFunction);
    }
    for (let conf of confs) {
        conf.noUiSlider.on('update',updFunction);
    }
}

function htmlToElement(html) {
    var template = document.createElement('template');
    html = html.trim(); // Never return a text node of whitespace as the result
    template.innerHTML = html;
    return template.content.firstChild;
}

function addReview() {
    let reviews = document.getElementById('reviews');
    let newReview = htmlToElement(' <div class="flex-container"> <div class="rating flex-child"></div> <div class="confidence flex-child"></div> </div>');
    reviews.appendChild(document.createTextNode('Review ' + (ratings.length + 1) + ': '));
    reviews.appendChild(newReview);
    reviews.appendChild(htmlToElement('<br>'));
    reviews.appendChild(htmlToElement('<br>'));
    initSliders();

}

function updatePage(prob) {
    let disp_prob = document.getElementById('accept_prob');
    disp_prob.innerHTML = (prob * 100).toFixed(2) + '%';
    let disp_text = document.getElementById('accept_text');
    if (prob > 0.9) {
        disp_text.innerHTML = "Congrats!";
    } else if (prob > 0.6) {
        disp_text.innerHTML = "Odds are looking good!";
    } else if (prob > 0.4) {
        disp_text.innerHTML = "It's pretty much a coin flip...";
    } else if (prob > 0.1) {
        disp_text.innerHTML = "Hope for the AC to come through :'(";
    } else {
        disp_text.innerHTML = "Better luck next conference!";
    }
}

initSliders();


