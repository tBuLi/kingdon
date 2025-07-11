const Algebra = await fetch("https://enki.ws/ganja.js/ganja.js")
                      .then(x=>x.text())
                      .then(x=>{ const ctx = {}; (new Function('const define=1;'+x)).apply(ctx); return ctx.Algebra });

function render({ model, el }) {
    var canvas = Algebra({metric: model.get('signature')}).inline((model)=>{
        // Define constants
        var key2idx = model.get('key2idx');
        var draggable_points_idxs = model.get('draggable_points_idxs');
        var options = model.get('options');
        var d = model.get('signature').length;

        function grade(key) {
            var count = 0;
            while (key) {
                count += key & 1;
                key >>= 1;
            }
            return count;
        }

        // Define helper functions.
        var toElement = (o)=>{
            /* convert object to Element */
            var _values = o['mv'] instanceof DataView?new Float64Array(o['mv'].buffer):o['mv'];
            if ('grades' in o) {
                var values = new Element();
                o['grades'].forEach(grade=>values[grade] = []);
                o['keys'].forEach((k, j)=>{var g = grade(k); values[g][key2idx[g][k]] = _values[j]});
                return values;
            }
            if ('keys' in o) {
                var values = Array(Object.keys(key2idx).length).fill(0);
                o['keys'].forEach((k, j)=>values[key2idx[k]] = _values[j]);
                return new Element(values);
            }
            return new Element(_values);
        }
        var decode = x=>typeof x === 'object' && 'mv' in x?toElement(x):Array.isArray(x)?x.map(decode):x;
        var encode = x=>x instanceof Element?({mv:[...x]}):x?.map?x.map(encode):x;

        // Decode camera if provided.
         if (options?.camera && typeof options.camera === 'object' && 'mv' in options.camera) {
             options.camera = toElement(options.camera)
         }

        if (options?.animate) {
            var graph_func = ()=>{
                if (canvas?.value && draggable_points_idxs?.length) {
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
                if (canvas?.value && draggable_points_idxs?.length) {
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
