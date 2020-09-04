
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
let fullModel = {
    lin1: tf.tensor([[-0.2452, 0.1776],
    [0.8782, -0.4691],
    [0.8423, -0.2451],
    [-0.5410, 0.5914],
    [-0.8162, -0.1508],
    [0.2367, 0.0086],
    [0.9297, -0.1585],
    [0.5735, 0.3087],
    [0.7918, -0.1871],
    [-0.1215, 0.0740],
    [-0.6670, 0.6221],
    [-0.9716, 0.3117],
    [-0.5104, -0.4767],
    [-0.2594, -0.7813],
    [0.7132, 0.4168],
    [-0.5183, -0.6321],
    [0.4532, -0.3750],
    [-0.5699, 0.3726],
    [0.6322, -0.2175],
    [0.7871, -0.0623],
    [0.7866, -0.0792],
    [0.8779, 0.2440],
    [-0.3661, 0.3980],
    [0.6781, -0.4053],
    [0.3276, 0.0829]]), lin1_bias: tf.tensor([0.7070, -0.3538, 0.3259, 0.0037, 0.0752, -0.6683, -0.2003, 0.3196,
        0.0667, -0.5051, -0.0536, 0.4716, 0.1371, 0.8101, 0.4177, -0.1057,
        -0.2481, 0.0226, -0.0101, -0.2215, -0.2732, -0.0483, 0.1117, -0.2737,
        -0.5890]), lin2: tf.tensor([[0.2269, -0.2905, -0.3318, 0.3023, 0.0336, 0.1072, -0.2723, -0.2431,
            -0.1504, -0.1832, 0.2540, 0.3430, 0.3008, 0.1033, -0.1363, 0.2912,
            -0.0062, 0.3716, -0.0911, -0.3765, -0.2824, -0.3078, 0.4614, -0.2583,
            0.0463],
        [-0.2702, 0.0105, 0.4035, -0.2453, -0.2687, -0.0329, 0.2157, 0.1413,
            0.2884, -0.0041, -0.1289, -0.0709, -0.2909, -0.2973, 0.2175, -0.3471,
            0.1323, -0.3440, 0.0720, 0.3093, 0.0502, 0.3133, -0.2455, 0.2585,
        -0.0952]]), lin2_bias: tf.tensor([0.3264, -0.2709]),
    mean: tf.tensor([5.44976315, 3.76464722]),
    variance: tf.tensor([2.19262709, 0.67709473]),
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
        const prob = (get_prob(data, closeModel) + get_prob(data, fullModel))/2;
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


