
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
    for (let rating of ratings) {
        rating.noUiSlider.on('update', function () {
            const prob = get_prob(getData(), closeModel);
            updatePage(prob);
        });
    }
    for (let conf of confs) {
        conf.noUiSlider.on('update', function () {
            const prob = get_prob(getData(), closeModel);
            updatePage(prob);
        });
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
    reviews.appendChild(document.createTextNode('Review ' + (ratings.length + 1) +': '));
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


