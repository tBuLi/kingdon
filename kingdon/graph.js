const Algebra = await fetch("https://enki.ws/ganja.js/ganja.js")
                      .then(x=>x.text())
                      .then(x=>{ const ctx = {}; (new Function('const define=1;'+x)).apply(ctx); return ctx.Algebra });

function render({ model, el }) {
    var canvas = Algebra({metric: model.get('signature')}).inline((model)=>{
        // Define constants
        var key2idx = model.get('key2idx');
        var draggable_points_idxs = model.get('draggable_points_idxs');
        var options = model.get('options');

        // Define helper functions.
        var toElement = (o)=>{
            /* convert single mv object to Element */
            var _values = o['mv'] instanceof DataView?new Float64Array(o['mv'].buffer):o['mv'];
            if ('keys' in o) {
                var values = Array(Object.keys(key2idx).length).fill(0);
                o['keys'].forEach((k, j)=>values[key2idx[k]] = _values[j]);
                return new Element(values);
            }
            return new Element(_values);
        }
        var toArray = (o)=>{
            /* convert mv object with shape to array of objects representing single mv's. */
            var shape = o['shape'];
            if (o['mv'] instanceof DataView) {
                var _values = new Float64Array(o['mv'].buffer);
                var objects = [];
                // TODO: allow higher dimenional shapes.
                for (let i = 0; i < shape[1]; i++) {
                    var obj = {'keys': o['keys'], 'mv': []}
                    for (let j = 0; j < shape[0]; j++) {
                        obj['mv'].push(_values[j*shape[1] + i]);
                    }
                    objects.push(obj);
                }
            } else {
                var _values = o['mv'].map(x=>new Float64Array(x.buffer));
                var objects = [];
                for (let i = 0; i < shape[1]; i++) {
                    objects.push({'mv': _values.map(x=>x[i]), 'keys': o['keys']});
                }
            }
            return objects
        }
        var decode = (input)=>{
            var output = [];
            for (let i=0; i<input.length; i++) {
                var x = input[i];
                if (typeof x === 'object' && 'shape' in x) {
                    output = [...output, ...(toArray(x).map(toElement))];
                } else if (typeof x === 'object' && 'mv' in x) {
                    output.push(toElement(x));
                } else if (Array.isArray(x)) {
                    output.push(x.map(toElement));
                } else {
                    output.push(x);
                }
            }
            return output;
        }
        var encode = x=>x instanceof Element?({mv:[...x]}):x.map?x.map(encode):x;

        // Decode camera if provided.
         if (options?.camera) {
             options.camera = toElement(options.camera)
         }

        if (options?.animate) {
            var graph_func = ()=>{
                if (canvas?.value) {
                    model.set('draggable_points', encode(draggable_points_idxs.map(i=>canvas.value[i])));
                    model.save_changes();
                }
                // Send an update request. This drives the event loop.
                model.send({ type: "update_mvs" });
                var subjects = decode(model.get('subjects'));
                return [...subjects];
            }
        } else {
            var graph_func = ()=>{
                if (canvas?.value) {
                    model.set('draggable_points', encode(draggable_points_idxs.map(i=>canvas.value[i])));
                    model.save_changes();
                }
                var subjects = decode(model.get('subjects'));
                return [...subjects];
            }

            // This ensures the remake is always called one last time to show the final position.
            model.on("change:subjects", ()=>{
                if (canvas.remake) canvas = canvas.remake(0);
                if (canvas.update) canvas.update(canvas.value);
            });
        }

        var canvas;
        canvas = this.graph(graph_func, options)
        return canvas;
    })(model)

    var options = model.get('options');
    canvas.style.width = options?.width || `min( 100%, 1024px )`;
    canvas.style.height = options?.height || 'auto';
    canvas.style.aspectRatio = '16 / 6';
    canvas.style.background = 'white';
    canvas.style.marginLeft = `calc( (100% - ${ options?.width??"min(100%, 1024px)" }) / 2 )`;
    el.appendChild(canvas);
}

export default { render };
